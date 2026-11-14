import torch
import torch.nn.functional as F

from .chemistry.routes import MAX_REACTIONS, MIN_REACTIONS


def rollout(
    model,
    env,
    building_blocks,
    config,
    target_features,
    starts,
    *,
    training=False,
    reference=None,
):
    """Run a batch under the truncated synthesis policy and retain completed reactions."""
    state = env.reset(target_features, starts)
    trajectory = {"log_probs": [], "entropies": [], "embedding_mses": []}
    terminations = ["truncated_end"] * config.batch_size
    allowed = torch.tensor(
        [[True, False, True] if smiles else [False, True, False] for smiles in starts],
        dtype=torch.bool,
        device=config.device,
    )
    end_only = torch.tensor([False, False, True], device=config.device)

    for step in range(config.max_seq_len):
        reaction_counts = [len(route.steps) for route in env.routes]
        capped = [
            active and count >= MAX_REACTIONS
            for active, count in zip(env.active, reaction_counts)
        ]
        allowed[torch.tensor(capped, device=config.device)] = end_only
        if training:
            short = [
                active and count < MIN_REACTIONS
                for active, count in zip(env.active, reaction_counts)
            ]
            allowed[torch.tensor(short, device=config.device), 2] = False

        action_logits, template_logits, fp_preds, features = model.generate_step(state)
        if reference is not None:
            with torch.no_grad():
                _, _, _, reference_features = reference.generate_step(state)
            trajectory["embedding_mses"].append(
                F.mse_loss(features, reference_features, reduction="none").mean(dim=-1)
            )

        action_dict, log_prob, entropy = model.sample_and_compute_logprob(
            action_logits,
            template_logits,
            fp_preds,
            features,
            allow_action=allowed,
            faiss_manager=building_blocks,
            sim2fp_fn=env.fingerprints.from_smiles,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k_0 if step == 0 else config.top_k,
            template_validator=env.check_template,
            reactant_validator=env.check_reactant,
        )
        if training:
            trajectory["log_probs"].append(log_prob)
            trajectory["entropies"].append(entropy)

        actions = action_dict["action"].tolist()
        active_before = env.active.copy()
        state = env.step(
            actions,
            action_dict["template"].tolist(),
            action_dict["sampled_smiles"],
            action_dict["sampled_real_fps"],
        )
        for i, action in enumerate(actions):
            if active_before[i] and not env.active[i]:
                if action == 2:
                    terminations[i] = "normal_end"
                elif len(env.routes[i].steps) < MIN_REACTIONS:
                    terminations[i] = (
                        "template_fail" if action == 0 else "reactant_fail"
                    )

        if not any(env.active):
            break
        allowed = action_dict["next_allow_action"]

    return trajectory, terminations
