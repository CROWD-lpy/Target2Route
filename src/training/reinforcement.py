import random
import time
from collections import Counter, defaultdict, deque
from copy import deepcopy

import torch

from ..chemistry.building_blocks import BuildingBlockIndex
from ..chemistry.docking import docking_pool
from ..chemistry.environment import BatchChemEnv
from ..chemistry.routes import (
    SUCCESSFUL_ENDS,
    VALID_OUTPUT_FATES,
    route_record,
    write_routes,
)
from ..rollout import rollout
from ..runtime import load_pickle, model_from_checkpoint, save_torch, setup_logger
from .rewards import build_catalog, score_routes


class RLTrainer:
    def __init__(self, config, checkpoint, run_dir, targets, pool):
        self.config = config
        self.run_dir = run_dir
        self.targets = targets
        self.pool = pool
        self.logger = setup_logger(run_dir, "train_rl.log")
        self.logger.info("Loading %s for RL with %d target(s)", checkpoint, len(pool))
        self.model, _ = model_from_checkpoint(config, checkpoint)
        self.reference = None
        if config.emb_mse_coef:
            self.reference = deepcopy(self.model)
            self.reference.requires_grad_(False).eval()
        if config.prediction_head_freeze:
            self.model.freeze_prediction_heads()
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.env = BatchChemEnv(config)
        self.building_blocks = BuildingBlockIndex(config)
        self.catalog = build_catalog()
        self.target_fps = defaultdict(
            lambda: deque(maxlen=config.per_target_buffer_size)
        )
        self.train_smiles = (
            {record["target_smi"] for record in load_pickle(config.train_data_path)}
            if config.train_data_path
            else None
        )
        if len(pool) == 1:
            # Resolve a missing sequence once before starting the rollout loop.
            self.targets.features.get(pool[0].sequence)

    def _starts(self, targets):
        denovo, scaffold = self.config.ratio_denovo_scaffold
        denovo_count = int(len(targets) * denovo / (denovo + scaffold))
        modes = ["denovo"] * denovo_count + ["scaffold"] * (len(targets) - denovo_count)
        random.shuffle(modes)
        starts = [""] * len(targets)
        for i, (target, mode) in enumerate(zip(targets, modes)):
            if mode != "scaffold" or not target.start_mols:
                continue
            anchor = random.choice(target.start_mols)
            fp = self.env.fingerprints.from_smiles(anchor)
            hits = self.building_blocks.search_continuous(
                fp, top_k=self.config.top_k_0, nprobe=64
            )
            if hits:
                starts[i] = random.choice(hits)[0]
        return modes, starts

    def _optimize_policy(self, trajectory, rewards):
        config = self.config
        rewards = rewards.to(config.device)
        log_probs = torch.stack(trajectory["log_probs"], dim=1).sum(dim=1)
        entropies = torch.stack(trajectory["entropies"], dim=1).mean(dim=1)
        advantages = rewards - torch.quantile(rewards, config.quantile_q)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -config.entropy_coef * entropies.mean()
        mse_loss = policy_loss.new_zeros(())
        if self.reference is not None:
            mse_loss = (
                config.emb_mse_coef
                * torch.stack(trajectory["embedding_mses"], dim=1).sum(dim=1).mean()
            )

        loss = policy_loss + entropy_loss + mse_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), policy_loss.item(), entropy_loss.item(), mse_loss.item()

    def _report(
        self, writer, episode, targets, modes, terminations, scores, losses, elapsed
    ):
        successful = [
            i for i, fate in enumerate(scores.fates) if fate in SUCCESSFUL_ENDS
        ]
        valid = [i for i, fate in enumerate(scores.fates) if fate in VALID_OUTPUT_FATES]
        records = {
            i: route_record(
                episode,
                modes[i],
                targets[i],
                self.env.routes[i],
                terminations[i],
                scores,
                i,
            )
            for i in valid
        }
        if valid:
            write_routes(self.run_dir / "routes_valid.csv", (records[i] for i in valid))
        if successful:
            write_routes(
                self.run_dir / "routes_reward.csv", (records[i] for i in successful)
            )

        def mean(values, indices):
            return values[indices].mean().item() if indices else 0.0

        metrics = {
            "RL_QED": scores.qed_scores,
            "RL_Vina": scores.vina_scores,
            "RL_SAScore": scores.sa_scores,
            "RL_RxnSteps": torch.tensor(
                [len(route.steps) for route in self.env.routes], dtype=torch.float32
            ),
        }
        for label, indices in (
            ("Overall", list(range(len(modes)))),
            ("Denovo", [i for i, mode in enumerate(modes) if mode == "denovo"]),
            ("Scaffold", [i for i, mode in enumerate(modes) if mode == "scaffold"]),
        ):
            succeeded = sorted(set(indices).intersection(successful))
            writer.add_scalar(
                f"RL_Reward/{label}", mean(scores.rewards, indices), episode
            )
            # The original per-mode dashboards tracked template/reactant completion.
            count = (
                len(succeeded)
                if label == "Overall"
                else sum(
                    scores.fates[i] not in {"template_fail", "reactant_fail"}
                    for i in indices
                )
            )
            writer.add_scalar(
                f"RL_SuccessRate/{label}",
                count / len(indices) if indices else 0.0,
                episode,
            )
            for metric, values in metrics.items():
                writer.add_scalar(f"{metric}/{label}", mean(values, succeeded), episode)

        loss, policy, entropy, mse = losses
        writer.add_scalar("RL/Loss", loss, episode)
        writer.add_scalar("RL/Loss_Emb_MSE", mse, episode)
        writer.add_scalar(
            "RL_Diversity/InternalDiversity", scores.internal_diversity, episode
        )
        writer.add_scalar(
            "RL_Diversity/ScaffoldDiversity", scores.scaffold_diversity, episode
        )
        writer.add_scalar("RL_Diversity/NoveltyRate", scores.novelty_rate, episode)
        counts = Counter(scores.fates)
        for fate, count in counts.items():
            writer.add_scalar(f"Fates/{fate.capitalize()}", count, episode)
        self.logger.info(
            "Episode %d/%d %.1fs | loss %.4f (policy %.4f, entropy %.4f, embedding %.4f) "
            "| reward %.4f | QED %.3f | Vina %.3f | outcomes %s",
            episode,
            self.config.num_episodes,
            elapsed,
            loss,
            policy,
            entropy,
            mse,
            scores.rewards.mean().item(),
            mean(scores.qed_scores, successful),
            mean(scores.vina_scores, successful),
            dict(counts),
        )

    def train(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "Install TensorBoard with pip install -e '.[training]'"
            ) from exc

        cfg = self.config
        checkpoint_dir = self.run_dir / "ckpt"
        checkpoint_dir.mkdir()
        write_routes(self.run_dir / "routes_valid.csv")
        write_routes(self.run_dir / "routes_reward.csv")
        with (
            SummaryWriter(log_dir=str(self.run_dir / "tensorboard")) as writer,
            docking_pool(cfg) as dock,
        ):
            for episode in range(1, cfg.num_episodes + 1):
                start_time = time.perf_counter()
                self.model.train()
                targets, features = self.targets.batch(self.pool)
                modes, starts = self._starts(targets)
                trajectory, terminations = rollout(
                    self.model,
                    self.env,
                    self.building_blocks,
                    cfg,
                    features,
                    starts,
                    training=True,
                    reference=self.reference,
                )
                scores = score_routes(
                    cfg,
                    self.env.routes,
                    terminations,
                    targets,
                    dock,
                    self.env.fingerprints,
                    self.catalog,
                    self.target_fps,
                    self.train_smiles,
                )
                losses = self._optimize_policy(trajectory, scores.rewards)
                self._report(
                    writer,
                    episode,
                    targets,
                    modes,
                    terminations,
                    scores,
                    losses,
                    time.perf_counter() - start_time,
                )
                if episode % cfg.save_freq == 0 or episode == cfg.num_episodes:
                    path = checkpoint_dir / f"rl_model_ep_{episode}.pt"
                    save_torch(
                        {
                            "episode": episode,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        path,
                    )
                    self.logger.info("Saved %s", path)
