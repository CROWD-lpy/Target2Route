from collections import Counter
from dataclasses import dataclass

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, FilterCatalog, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..chemistry import sascorer
from ..chemistry.routes import MIN_REACTIONS, SUCCESSFUL_ENDS


def build_catalog():
    params = FilterCatalog.FilterCatalogParams()
    for name in ("PAINS", "BRENK", "NIH"):
        params.AddCatalog(getattr(params.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


@dataclass
class RewardScores:
    fates: list[str]
    rewards: torch.Tensor
    qed_scores: torch.Tensor
    vina_scores: torch.Tensor
    sa_scores: torch.Tensor
    max_sims: torch.Tensor
    sim_penalties: torch.Tensor
    template_penalties: torch.Tensor
    internal_diversity: float
    scaffold_diversity: float
    novelty_rate: float


def _evaluate_molecule_properties(smi, config, catalog):
    """Return the properties of a molecule and whether it passes the configured filters."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, None

    try:
        props = {
            "qed": QED.default(mol),
            "logp": Descriptors.MolLogP(mol),
            "mw": Descriptors.MolWt(mol),
            "hbd": rdMolDescriptors.CalcNumLipinskiHBD(mol),
            "hba": rdMolDescriptors.CalcNumLipinskiHBA(mol),
            "nrotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
        }
        group_pass = not catalog.HasMatch(mol) if catalog else True
    except (ValueError, RuntimeError):
        return False, None

    limits = {
        "min_qed": getattr(config, "hard_min_qed", None),
        "min_logp": getattr(config, "hard_min_logp", None),
        "max_logp": getattr(config, "hard_max_logp", None),
        "min_mw": getattr(config, "hard_min_mw", None),
        "max_mw": getattr(config, "hard_max_mw", None),
        "max_hbd": getattr(config, "hard_max_hydrogen_donor", None),
        "max_hba": getattr(config, "hard_max_hydrogen_acceptor", None),
        "max_nrotb": getattr(config, "hard_max_nrotb", None),
        "max_tpsa": getattr(config, "hard_max_tpsa", None),
    }

    checks = [
        group_pass,
        limits["min_qed"] is None or props["qed"] >= limits["min_qed"],
        limits["min_logp"] is None or props["logp"] >= limits["min_logp"],
        limits["max_logp"] is None or props["logp"] <= limits["max_logp"],
        limits["min_mw"] is None or props["mw"] >= limits["min_mw"],
        limits["max_mw"] is None or props["mw"] <= limits["max_mw"],
        limits["max_hbd"] is None or props["hbd"] <= limits["max_hbd"],
        limits["max_hba"] is None or props["hba"] <= limits["max_hba"],
        limits["max_nrotb"] is None or props["nrotb"] <= limits["max_nrotb"],
        limits["max_tpsa"] is None or props["tpsa"] <= limits["max_tpsa"],
    ]

    return all(checks), props


def _calculate_combined_score(config, v_score, qval):
    if config.reward_method == "linear_product":
        return -1.0 * v_score * qval
    elif config.reward_method == "quadratic_baseline":
        return qval * max(0, -v_score + config.vina_base) ** 2
    return 0.0


def _apply_diversity_penalty(
    config, i, fp_i, target_idx, valid_fps_dict, target_fp_buffers, rewards
):
    """Discount rewards for high fingerprint similarity within a target."""
    max_sim = 0.0

    batch_other_fps = [
        data["fp"]
        for j, data in valid_fps_dict.items()
        if j != i and data["target_idx"] == target_idx
    ]
    if batch_other_fps:
        max_sim = max(
            max_sim, max(DataStructs.BulkTanimotoSimilarity(fp_i, batch_other_fps))
        )

    use_history = getattr(config, "use_historical_penalty", False)
    if use_history and target_fp_buffers is not None:
        target_buffer = target_fp_buffers[target_idx]
        if target_buffer:
            max_sim = max(
                max_sim,
                max(DataStructs.BulkTanimotoSimilarity(fp_i, list(target_buffer))),
            )

    penalty_ratio = 0.0
    if max_sim > config.sim_threshold:
        penalty_ratio = (max_sim - config.sim_threshold) / (1.0 - config.sim_threshold)
        penalty_ratio = min(max(penalty_ratio, 0.0), 1.0)
        rewards[i] *= 1.0 - config.max_discount_ratio * penalty_ratio

    return rewards[i], max_sim, penalty_ratio


def _apply_template_penalty(config, routes, fates, rewards):
    # Keep the original reward convention, which excludes template ID 0.
    used = {
        i: [step.template_id for step in route.steps if step.template_id != 0]
        for i, route in enumerate(routes)
        if fates[i] in SUCCESSFUL_ENDS and rewards[i] > 0
    }
    counts = Counter(template for steps in used.values() for template in steps)
    denominator = max(sum(counts.values()), 40)
    threshold = getattr(config, "template_freq_threshold", 0.10)
    factor = getattr(config, "template_penalty_factor", 10.0)
    penalties = torch.zeros_like(rewards)
    for i, steps in used.items():
        penalties[i] = sum(
            max(0.0, counts[template] / denominator - threshold) * factor
            for template in steps
        )
    rewards -= penalties
    return penalties


def _compute_diversity_novelty_metrics(
    valid_fps_dict, best_smiles_dict, train_set_smiles=None
):
    """Calculate batch fingerprint diversity, scaffold diversity and training-set novelty."""
    fps_list = [data["fp"] for data in valid_fps_dict.values()]
    indices = list(valid_fps_dict.keys())
    n = len(fps_list)

    internal_diversity = 0.0
    if n >= 2:
        total_sim, count = 0.0, 0
        for a in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps_list[a], fps_list[a + 1 :])
            total_sim += sum(sims)
            count += len(sims)
        internal_diversity = 1.0 - total_sim / count if count > 0 else 0.0

    scaffold_diversity = 0.0
    if n > 0:
        unique_scaffolds = set()
        for i in indices:
            smi = best_smiles_dict.get(i)
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    try:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                            mol=mol, includeChirality=False
                        )
                        unique_scaffolds.add(scaffold)
                    except (ValueError, RuntimeError):
                        pass
        scaffold_diversity = len(unique_scaffolds) / n

    novelty_rate = 0.0
    if n > 0:
        if train_set_smiles is not None:
            novel_count = 0
            for i in indices:
                smi = best_smiles_dict.get(i)
                if smi and smi not in train_set_smiles:
                    novel_count += 1
            novelty_rate = novel_count / n
        else:
            novelty_rate = 0.0

    return internal_diversity, scaffold_diversity, novelty_rate


def score_routes(
    config,
    routes,
    terminations,
    targets,
    dock,
    fingerprints,
    catalog,
    target_fp_buffers=None,
    train_set_smiles=None,
):
    batch_size = len(routes)
    fates = list(terminations)
    rewards = torch.full((batch_size,), config.penalty_reward, dtype=torch.float32)
    qed_scores = torch.zeros(batch_size)
    vina_scores = torch.zeros(batch_size)
    sa_scores = torch.full((batch_size,), 10.0)
    docking_jobs = {}

    for i, (route, target) in enumerate(zip(routes, targets)):
        if fates[i] == "reactant_fail" and config.penalty_staged:
            rewards[i] -= 1.0
        if fates[i] not in SUCCESSFUL_ENDS:
            continue
        if len(route.steps) < MIN_REACTIONS:
            fates[i] = "short_route_fail"
            continue

        valid, properties = _evaluate_molecule_properties(
            route.product, config, catalog
        )
        if not valid:
            fates[i] = "threshold_fail"
            if config.penalty_staged:
                rewards[i] += 1.0
            continue
        job = (str(target.receptor), target.box, route.product)
        docking_jobs[i] = (job, properties["qed"])

    unique_jobs = list(dict.fromkeys(job for job, _ in docking_jobs.values()))
    docking_scores = dock(unique_jobs) if unique_jobs else {}
    best_smiles = {}
    valid_fps = {}
    for i, (job, qed) in docking_jobs.items():
        vina = docking_scores[job]
        if vina is None:
            fates[i] = "docking_fail"
            continue
        combined = _calculate_combined_score(config, vina, qed)
        if combined <= 0:
            fates[i] = "threshold_fail"
            continue

        qed_scores[i], vina_scores[i] = qed, vina
        smiles = routes[i].product
        best_smiles[i] = smiles
        mol = Chem.MolFromSmiles(smiles)
        sa_scores[i] = sascorer.calculateScore(mol)
        length_bonus = (len(routes[i].steps) - MIN_REACTIONS) * getattr(
            config, "step_reward", 0.0
        )
        trunc_penalty = (
            getattr(config, "penalty_trunc", -0.5)
            if fates[i] == "truncated_end"
            else 0.0
        )
        rewards[i] = combined + length_bonus + trunc_penalty
        if rewards[i] > 0:
            valid_fps[i] = {
                "fp": fingerprints.bit_vector(mol),
                "target_idx": targets[i].id,
            }

    max_sims, sim_penalties = torch.zeros(batch_size), torch.zeros(batch_size)
    for i, data in valid_fps.items():
        fp, target_id = data["fp"], data["target_idx"]
        rewards[i], max_sims[i], sim_penalties[i] = _apply_diversity_penalty(
            config, i, fp, target_id, valid_fps, target_fp_buffers, rewards
        )
        if target_fp_buffers is not None:
            target_fp_buffers[target_id].append(fp)

    template_penalties = _apply_template_penalty(config, routes, fates, rewards)
    diversity, scaffold_diversity, novelty = _compute_diversity_novelty_metrics(
        valid_fps, best_smiles, train_set_smiles
    )
    return RewardScores(
        fates,
        rewards,
        qed_scores,
        vina_scores,
        sa_scores,
        max_sims,
        sim_penalties,
        template_penalties,
        diversity,
        scaffold_diversity,
        novelty,
    )
