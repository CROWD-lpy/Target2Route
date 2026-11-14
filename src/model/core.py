import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from ..chemistry.templates import load_templates
from .diffusion.model_diffusion import MlpDiffusionFingerprintHead
from .sampling import top_p_filtering


class Target2RouteGen(nn.Module):
    def __init__(self, config, *, fp_pos_weight=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        pos_weight = fp_pos_weight
        if pos_weight is None:
            if isinstance(config.fp_pos_weights, str):
                weight_table = torch.load(
                    config.fp_pos_weights, map_location="cpu", weights_only=True
                )
                pos_weight = weight_table[f"{config.fp_type.lower()}_{config.fp_dim}"]
            else:
                pos_weight = torch.tensor([float(config.fp_pos_weights)])
        if pos_weight.shape not in ((1,), (config.fp_dim,)):
            raise ValueError(
                f"Invalid fp_pos_weight shape {pos_weight.shape} for fp_dim {config.fp_dim}"
            )
        self.register_buffer("fp_pos_weight", pos_weight)

        self.target_proj = nn.Sequential(
            nn.Linear(config.esm2_dim + config.pocket_dim, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.start_proj = nn.Sequential(
            nn.Linear(config.fp_dim, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )

        self.pos_embed = nn.Embedding(self.config.max_seq_len + 2, self.hidden_size)
        self.template_embed = nn.Embedding(config.num_templates, self.hidden_size)
        self.fp_proj = nn.Sequential(
            nn.Linear(config.fp_dim, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.token_type_embed = nn.Embedding(
            3, self.hidden_size
        )  # 0: Template, 1: Reactant, 2: End

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )

        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 3),
            nn.GELU(),
            nn.Linear(self.hidden_size * 3, 3),
        )
        self.template_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 3),
            nn.GELU(),
            nn.Linear(self.hidden_size * 3, config.num_templates),
        )
        if config.fp_head == "mlp":
            self.fp_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 3),
                nn.GELU(),
                nn.Linear(self.hidden_size * 3, config.fp_dim),
            )
        elif config.fp_head == "diffusion":
            self.fp_head = MlpDiffusionFingerprintHead(
                pos_weight=self.fp_pos_weight,
                d_model=config.hidden_size,
                fingerprint_dim=config.fp_dim,
                hidden_dim=config.hidden_size,
                diffusion_steps=config.diffusion_steps,
                s=config.diffusion_s,
            )
            self.num_timestep_samples = config.num_timestep_samples
        else:
            raise ValueError(f"Unsupported fp_head: {config.fp_head}")

        templates = load_templates(config)
        self.register_buffer(
            "is_bimol", torch.tensor((templates["num_reactants"] == 2).to_numpy())
        )

    def _batch_to_device(self, batch):
        device = self.config.device
        batch.target_features = batch.target_features.to(device)
        batch.start_features = batch.start_features.to(device)
        batch.seq_types = batch.seq_types.to(device)
        batch.seq_templates = batch.seq_templates.to(device)
        batch.seq_fps = batch.seq_fps.to(device)
        return batch

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
        )
        return mask

    def forward(self, batch):
        """
        batch.target_features: [bsz, esm2_dim + pocket_dim]
        batch.start_features: [bsz, fp_dim]
        batch.seq_types: [bsz, seq_len] (0: template, 1: reactant, 2: end)
        batch.seq_templates: [bsz, seq_len]
        batch.seq_fps: [bsz, seq_len, fp_dim]
        """
        batch = self._batch_to_device(batch)
        device = batch.target_features.device
        target_emb = self.target_proj(batch.target_features).unsqueeze(1)  # [B, 1, H]
        start_emb = self.start_proj(batch.start_features).unsqueeze(1)

        t_mask = (batch.seq_types == 0).unsqueeze(-1).float()  # [B, seq_len, 1]
        r_mask = (batch.seq_types == 1).unsqueeze(-1).float()  # [B, seq_len, 1]
        e_mask = (batch.seq_types == 2).unsqueeze(-1).float()
        t_emb = self.template_embed(batch.seq_templates) + self.token_type_embed(
            torch.tensor(0, device=device)
        )
        r_emb = self.fp_proj(batch.seq_fps) + self.token_type_embed(
            torch.tensor(1, device=device)
        )
        e_emb = self.token_type_embed(torch.tensor(2, device=device))
        seq_emb = t_emb * t_mask + r_emb * r_mask + e_emb * e_mask  # [B, seq_len, H]

        full_seq = torch.cat(
            [target_emb, start_emb, seq_emb[:, :-1, :]], dim=1
        )  # Exclude the last teacher-forced token.
        total_len = full_seq.shape[1]
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        full_seq = full_seq + self.pos_embed(positions)
        mask = self.generate_square_subsequent_mask(total_len, device)

        hidden_states = self.transformer(full_seq, mask=mask, is_causal=True)
        pred_features = hidden_states[:, 1:, :]  # [B, seq_len, H]

        action_logits = self.action_head(pred_features)  # [B, seq_len, 3]
        template_logits = self.template_head(
            pred_features
        )  # [B, seq_len+1, num_templates]
        if self.config.fp_head == "mlp":
            fp_preds = self.fp_head(pred_features)  # [B, seq_len+1, fp_dim]
        else:
            fp_preds = None
        return action_logits, template_logits, fp_preds, pred_features

    def compute_loss(self, batch, mode):
        action_logits, template_logits, fp_preds, pred_features = self.forward(batch)

        pad_idx = -100
        valid_mask = batch.seq_types != pad_idx
        action_loss = F.cross_entropy(
            action_logits.transpose(1, 2),  # [B, 3, seq_len]
            batch.seq_types,  # [B, seq_len]
            ignore_index=pad_idx,
            reduction="mean",
        )
        action_preds = action_logits.argmax(dim=-1)
        if valid_mask.sum() > 0:
            action_acc = (
                (action_preds[valid_mask] == batch.seq_types[valid_mask]).float().mean()
            )
        else:
            action_acc = torch.tensor(0.0, device=action_logits.device)

        t_mask = batch.seq_types == 0
        if t_mask.sum() > 0:
            template_loss = F.cross_entropy(
                template_logits[t_mask],
                batch.seq_templates[t_mask],
                reduction="mean",
            )
            template_preds = template_logits[t_mask].argmax(dim=-1)
            template_acc = (
                (template_preds == batch.seq_templates[t_mask]).float().mean()
            )
        else:
            template_loss = torch.tensor(0.0, device=action_logits.device)
            template_acc = torch.tensor(0.0, device=action_logits.device)

        r_mask = batch.seq_types == 1
        if r_mask.sum() > 0:
            if self.config.fp_head == "mlp":
                fp_loss = F.binary_cross_entropy_with_logits(
                    fp_preds[r_mask],
                    batch.seq_fps[r_mask].float(),
                    pos_weight=self.fp_pos_weight,
                    reduction="mean",
                )
                fp_preds = torch.sigmoid(fp_preds[r_mask])
            else:
                fp_loss = self.fp_head.get_loss(
                    pred_features[r_mask],
                    batch.seq_fps[r_mask].float(),
                    self.num_timestep_samples,
                )
                if mode == "test":
                    test_try = getattr(self.config, "test_try", 1)
                    if test_try > 1:
                        n_samples = r_mask.sum()
                        expanded_features = pred_features[r_mask].repeat_interleave(
                            test_try, dim=0
                        )
                        expanded_fp_preds = self.fp_head.predict_probability(
                            expanded_features
                        )
                        fp_preds = expanded_fp_preds.view(n_samples, test_try, -1)
                    else:
                        fp_preds = self.fp_head.predict_probability(
                            pred_features[r_mask]
                        )
        else:
            fp_loss = torch.tensor(0.0, device=action_logits.device)
            fp_preds = None

        total_loss = (
            self.config.loss_ratio[0] * action_loss
            + self.config.loss_ratio[1] * template_loss
            + self.config.loss_ratio[2] * fp_loss
        )
        loss_dict = {
            "total_loss": total_loss,
            "action_loss": action_loss.detach(),
            "template_loss": template_loss.detach(),
            "fp_loss": fp_loss.detach(),
            "action_acc": action_acc.detach(),
            "template_acc": template_acc.detach(),
            "fp_preds": fp_preds.detach() if fp_preds is not None else None,
            "fp_targets": batch.seq_fps[r_mask].detach() if r_mask.sum() > 0 else None,
        }
        return total_loss, loss_dict

    def generate_step(self, states):
        action_logits, template_logits, fp_preds, pred_features = self.forward(states)
        last_action_logits = action_logits[:, -1, :]  # [B, 3]
        last_template_logits = template_logits[:, -1, :]  # [B, num_templates]
        last_pred_features = pred_features[:, -1, :]  # [B, hidden_size]

        if fp_preds is not None:
            last_fp_preds = fp_preds[:, -1, :]  # [B, fp_dim]
        else:
            last_fp_preds = None
        return (
            last_action_logits,
            last_template_logits,
            last_fp_preds,
            last_pred_features,
        )

    def sample_and_compute_logprob(
        self,
        action_logits,
        template_logits,
        fp_preds,
        pred_features,
        allow_action,
        faiss_manager,
        sim2fp_fn,
        temperature,
        top_p=0.9,
        top_k=5,
        template_validator=None,
        reactant_validator=None,
    ):

        bsz = action_logits.shape[0]
        device = action_logits.device

        action_logits_masked = action_logits.clone()
        action_logits_masked[~allow_action] = -float("Inf")

        action_probs = F.softmax(action_logits_masked, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()  # [B]
        action_log_prob = action_dist.log_prob(action)  # [B]

        step_entropy = action_dist.entropy().clone()

        t_mask = action == 0
        sampled_templates = torch.full((bsz,), -1, dtype=torch.long, device=device)
        template_log_prob = torch.zeros(bsz, device=device)

        if t_mask.any():
            active_t_indices = t_mask.nonzero(as_tuple=True)[0].tolist()
            template_top_k = min(top_k, template_logits.shape[-1])
            topk_logits, topk_indices = torch.topk(
                template_logits, template_top_k, dim=-1
            )

            for batch_idx in active_t_indices:
                cand_indices = topk_indices[batch_idx]  # [top_k]
                cand_logits = topk_logits[batch_idx] / temperature

                valid_mask = torch.tensor(
                    [
                        template_validator is None or template_validator(batch_idx, idx)
                        for idx in cand_indices.tolist()
                    ],
                    dtype=torch.bool,
                    device=device,
                )

                if not valid_mask.any():
                    sampled_templates[batch_idx] = -1
                    continue

                cand_logits = cand_logits.masked_fill(~valid_mask, -float("Inf"))
                filtered_logits = top_p_filtering(
                    cand_logits.unsqueeze(0), top_p=top_p
                ).squeeze(0)

                t_probs = F.softmax(filtered_logits, dim=-1)
                t_dist = Categorical(t_probs)
                sampled_local_idx = t_dist.sample()

                template_log_prob[batch_idx] = t_dist.log_prob(sampled_local_idx)
                sampled_templates[batch_idx] = cand_indices[sampled_local_idx]

                step_entropy[batch_idx] += t_dist.entropy()

        template = sampled_templates

        # FAISS distances and candidate selection are outside the gradient path.
        with torch.no_grad():
            if self.config.fp_head == "mlp":
                fp_cont = torch.sigmoid(fp_preds)
            else:
                fp_cont = self.fp_head.predict_probability(pred_features).squeeze(1)

        step_log_prob = action_log_prob.clone()
        t_mask = action == 0
        if t_mask.any():
            step_log_prob[t_mask] += template_log_prob[t_mask]

        r_mask = action == 1
        sampled_smiles_list = [None] * bsz
        sampled_real_fps = torch.zeros_like(fp_cont)

        if r_mask.any():
            search_fps_cpu = fp_cont[r_mask].detach().cpu()
            results = faiss_manager.search_continuous(
                search_fps_cpu, top_k=top_k, nprobe=64
            )

            active_r_indices = r_mask.nonzero(as_tuple=True)[0].tolist()
            for i, batch_idx in enumerate(active_r_indices):
                candidates = results[i]

                cand_mask = torch.zeros(
                    len(candidates), dtype=torch.bool, device=device
                )
                for c_idx, cand in enumerate(candidates):
                    smi = cand[0]
                    if reactant_validator is not None and reactant_validator(
                        batch_idx, smi
                    ):
                        cand_mask[c_idx] = True

                if not cand_mask.any():
                    sampled_smiles_list[batch_idx] = ""
                    continue
                distances = torch.tensor(
                    [cand[1] for cand in candidates], dtype=torch.float32, device=device
                )
                fp_logits = -distances / temperature
                fp_logits = fp_logits.masked_fill(~cand_mask, -float("inf"))
                filtered_fp_logits = top_p_filtering(
                    fp_logits.unsqueeze(0), top_p=top_p
                ).squeeze(0)

                fp_probs = F.softmax(filtered_fp_logits, dim=-1)
                fp_dist = Categorical(fp_probs)
                sampled_cand_idx = fp_dist.sample()

                step_log_prob[batch_idx] += fp_dist.log_prob(sampled_cand_idx)
                step_entropy[batch_idx] += fp_dist.entropy()

                sampled_smi = candidates[sampled_cand_idx.item()][0]
                sampled_smiles_list[batch_idx] = sampled_smi

                real_fp = sim2fp_fn(sampled_smi)
                sampled_real_fps[batch_idx] = torch.tensor(
                    real_fp, dtype=torch.float32, device=device
                )

        next_allow_action = torch.zeros((bsz, 3), dtype=torch.bool, device=device)
        if t_mask.any():
            safe_template = torch.where(
                template == -1, torch.zeros_like(template), template
            )
            is_bimol = self.is_bimol[safe_template]

            valid_t_mask = t_mask & (template != -1)
            failed_t_mask = t_mask & (template == -1)

            next_allow_action[valid_t_mask & is_bimol] = torch.tensor(
                [False, True, False], device=device
            )
            next_allow_action[valid_t_mask & ~is_bimol] = torch.tensor(
                [True, False, True], device=device
            )
            next_allow_action[failed_t_mask] = torch.tensor(
                [False, False, True], device=device
            )
        if r_mask.any():
            next_allow_action[r_mask] = torch.tensor([True, False, True], device=device)
        e_mask = action == 2
        if e_mask.any():
            next_allow_action[e_mask] = torch.tensor(
                [False, False, True], device=device
            )

        action_dict = {
            "action": action,
            "template": template,
            "sampled_smiles": sampled_smiles_list,
            "sampled_real_fps": sampled_real_fps,
            "next_allow_action": next_allow_action,
        }
        return action_dict, step_log_prob, step_entropy

    def freeze_prediction_heads(self):
        for head in (self.action_head, self.template_head, self.fp_head):
            head.requires_grad_(False)
