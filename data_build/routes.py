from collections import Counter
from functools import lru_cache

import pandas as pd
from rdkit import rdBase
from rdkit.Chem import Descriptors, rdChemReactions

from src.chemistry.routes import MAX_REACTIONS, MIN_REACTIONS
from src.esm2 import normalize_sequence

from .inputs import canonical_smiles, molecule
from .retrosynthesis import route_records


def _tree_steps(node):
    children = node.get("children", [])
    if not children:
        return []
    if len(children) != 1:
        return None
    reaction = children[0]
    reactants = reaction.get("children", [])
    if len(reactants) not in (1, 2):
        return None
    steps = []
    for child in reactants:
        preceding = _tree_steps(child)
        if preceding is None:
            return None
        steps.extend(preceding)
    meta = reaction.get("metadata", {})
    steps.append({
        "reactants": [child["smiles"] for child in reactants],
        "reactants_in_stock": [child.get("in_stock", False) for child in reactants],
        "template_code": meta.get("template_code"),
        "template_smiles": meta.get("template"),
        "product": node["smiles"],
    })
    return steps


@lru_cache(maxsize=50000)
def _mass(smiles):
    return Descriptors.MolWt(molecule(smiles))


def _linear_suffixes(steps, target, min_steps, max_steps, max_mass):
    for size in range(min(len(steps), max_steps), min_steps - 1, -1):
        suffix = steps[-size:]
        start = None
        stock_start = False
        current = None
        actions = []
        for step in suffix:
            reactants = [canonical_smiles(smi) for smi in step["reactants"]]
            stock = list(step["reactants_in_stock"])
            product = canonical_smiles(step["product"])
            if (len(reactants) not in (1, 2) or len(stock) != len(reactants)
                    or None in reactants or not product):
                break
            if start is None:
                # Prefer the unresolved intermediate as the starting condition.
                anchor = 1 if len(stock) == 2 and not stock[1] and stock[0] else 0
            else:
                choices = [
                    i for i, smi in enumerate(reactants)
                    if smi == current and all(flag for j, flag in enumerate(stock) if j != i)
                ]
                if not choices:
                    break
                anchor = choices[0]
            if not all(flag for j, flag in enumerate(stock) if j != anchor):
                break
            if start is None:
                start = reactants[anchor]
                stock_start = bool(stock[anchor])
                if _mass(start) > max_mass:
                    break
            template = step["template_code"]
            retro = step.get("template_smiles")
            if template is None or not isinstance(retro, str) or ">>" not in retro:
                break
            side = reactants[1 - anchor] if len(reactants) == 2 else ""
            actions.append((str(template), retro, side, product))
            current = product
        else:
            if current == target:
                yield start, tuple(actions), stock_start


def curated_routes(paths, held_out, counts, *, min_steps=MIN_REACTIONS,
                   max_steps=MAX_REACTIONS, max_mass=350.0):
    sequences = {}
    for record in route_records(paths):
        counts["ligand_records"] += 1
        try:
            seq = normalize_sequence(record.get("target_seq", record.get("target")))
            ligand = canonical_smiles(record.get("target_smi", record.get("smiles")) or "")
            if not ligand:
                raise ValueError("invalid ligand")
        except ValueError:
            counts["invalid_ligand_records"] += 1
            continue
        if seq in held_out:
            counts["held_out_ligand_records"] += 1
            continue
        seq = sequences.setdefault(seq, seq)
        routes = (
            (_tree_steps(tree) for tree in record["trees"])
            if "trees" in record else (route["steps"] for route in record.get("routes", []))
        )
        for steps in routes:
            counts["raw_routes"] += 1
            if not steps:
                continue
            for i, (start, actions, stock_start) in enumerate(
                _linear_suffixes(steps, ligand, min_steps, max_steps, max_mass)
            ):
                counts["curated_routes"] += 1
                yield seq, ligand, start, actions, stock_start and i == 0, i == 0


def read_templates(path, *, fixed):
    if fixed:
        frame = pd.read_csv(path, dtype={"template_code": str})
        required = {"template_code", "retro_template", "forward_template", "num_reactants"}
        if not required.issubset(frame):
            raise ValueError(f"Frozen template bank {path} must contain {sorted(required)}")
        expected = list(range(len(frame)))
        if "Trace_Template_ID" in frame and frame["Trace_Template_ID"].tolist() != expected:
            raise ValueError(f"Template IDs must be zero-based row numbers in {path}")
    elif path.suffix in {".hdf5", ".h5", ".hdf"}:
        frame = pd.read_hdf(path, "table")
    else:
        frame = pd.read_csv(path, sep="\t", index_col=0)
    if "retro_template" not in frame:
        raise ValueError(f"Missing retro_template in {path}")
    if not fixed:
        # AiZynthFinder records the template table's index as template_code.
        frame["template_code"] = frame.index.map(str)
    if frame.empty or frame["template_code"].isna().any() or frame["template_code"].duplicated().any():
        raise ValueError(f"Missing or duplicate template codes in {path}")
    frame["template_code"] = frame["template_code"].astype(str)
    return frame


def template_catalog(frame, *, fixed):
    catalog = {}
    columns = ["template_code", "retro_template"]
    if fixed:
        columns.extend(("forward_template", "num_reactants"))
    for row in frame[columns].itertuples(index=False, name=None):
        code, retro, *forward = row
        if not isinstance(retro, str) or retro.count(">>") != 1:
            continue
        product, precursors = retro.split(">>")
        smarts = forward[0] if fixed else f"{precursors}>>{product}"
        with rdBase.BlockLogs():
            try:
                reaction = rdChemReactions.ReactionFromSmarts(smarts)
            except (ValueError, RuntimeError):
                reaction = None
        if reaction is None:
            continue
        arity = reaction.GetNumReactantTemplates()
        if arity in (1, 2) and (not fixed or arity == forward[1]):
            catalog[str(code)] = (retro, smarts, arity)
    if fixed and len(catalog) != len(frame):
        raise ValueError("The frozen template bank contains unusable forward SMARTS or arities")
    return catalog


def select_templates(counts, coverage):
    if not counts:
        raise ValueError(
            "No usable template occurrences. Check the routes and the USPTO template release."
        )
    selected = set()
    threshold = coverage * sum(counts.values())
    cumulative = 0
    for code in sorted(counts, key=lambda key: (-counts[key], key)):
        selected.add(code)
        cumulative += counts[code]
        if cumulative >= threshold:
            break
    return selected, cumulative / sum(counts.values())


def build_routes(paths, options, config, held_out, template_frame, catalog):
    counts = Counter()
    frequencies = Counter()
    candidates = {}
    canonical_codes = {code: code for code in catalog}
    for seq, ligand, start, actions, denovo, longest in curated_routes(
        paths, held_out, counts, min_steps=options.min_steps,
        max_steps=options.max_steps, max_mass=options.start_max_mw,
    ):
        if any(code not in catalog or (retro, 1 + bool(side)) != (catalog[code][0], catalog[code][2])
               for code, retro, side, _ in actions):
            counts["template_mismatch"] += 1
            continue
        # Share template strings across routes and keep the ligand and sequence in
        # the identity. Merge evidence that a duplicated route is fully in stock.
        action_key = tuple((canonical_codes[code], side, product)
                           for code, _, side, product in actions)
        key = (seq, ligand, start, action_key)
        previous = candidates.get(key)
        if previous is not None:
            counts["duplicate_routes"] += 1
        if longest and (previous is None or not previous[1]):
            frequencies.update(code for code, *_ in actions)
        old_denovo, old_longest = previous or (False, False)
        candidates[key] = (old_denovo or denovo, old_longest or longest)
    if options.frozen_templates:
        order = template_frame["template_code"].tolist()
        codes = set(order)
        achieved = None
    else:
        codes, achieved = select_templates(frequencies, options.template_coverage)

    filtered = {}
    for key, (denovo, _) in candidates.items():
        actions = key[3]
        if any(code not in codes for code, *_ in actions):
            continue
        length = sum(1 + bool(side) for _, side, _ in actions)
        if length > config.max_seq_len:
            counts["overlength_examples"] += 1 + denovo
            continue
        if denovo and length == config.max_seq_len:
            counts["overlength_examples"] += 1
            denovo = False
        filtered[key] = denovo
    del candidates
    if not filtered:
        raise ValueError("No curated routes use only the selected TRACE templates")
    if not options.frozen_templates:
        surviving = {code for _, _, _, actions in filtered for code, *_ in actions}
        order = sorted(surviving, key=lambda key: (-frequencies[key], key))
    ids = {code: i for i, code in enumerate(order)}
    records = []
    for (seq, ligand, start, actions), denovo in filtered.items():
        route = []
        for code, side, product in actions:
            route.append(("T", ids[code], catalog[code][0]))
            if side:
                route.append(("R", side, product))
        records.append({
            "target_seq": seq, "target_smi": ligand, "start_mol": start,
            "task_prompt": "<SCAFFOLD>", "route": route,
        })
        if denovo:
            records.append({
                "target_seq": seq, "target_smi": ligand, "start_mol": "",
                "task_prompt": "<DE_NOVO>", "route": [("R", start, start), *route],
            })
    if not records:
        raise ValueError("No data fit max_seq_len in the model configuration")
    rows = template_frame.set_index("template_code").loc[order].reset_index()
    rows["Trace_Template_ID"] = range(len(rows))
    rows["forward_template"] = [catalog[code][1] for code in order]
    rows["num_reactants"] = [catalog[code][2] for code in order]
    counts["selected_routes"] = len(filtered)
    counts["supervised_examples"] = len(records)
    retained_frequency = sum(frequencies[code] for code in order)
    total_frequency = sum(frequencies.values())
    return records, rows, {
        "counts": dict(counts),
        "candidate_template_coverage": achieved,
        "retained_template_coverage": retained_frequency / total_frequency if total_frequency else None,
    }
