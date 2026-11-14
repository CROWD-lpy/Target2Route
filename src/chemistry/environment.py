import random

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import QED, rdChemReactions

from ..data import RouteBatch
from .fingerprints import Fingerprints
from .routes import ReactionStep, SynthesisRoute
from .templates import load_templates


class BatchChemEnv:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.device = config.device
        self.mode = config.rxn_mode
        self.fingerprints = Fingerprints(config)
        templates = load_templates(config)
        self.is_bimol = (templates["num_reactants"] == 2).tolist()
        self.reactions, self.usable_templates = [], []
        for index, (smarts, count) in enumerate(
            zip(templates["forward_template"], templates["num_reactants"])
        ):
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if rxn is None:
                raise ValueError(
                    f"Invalid reaction template at row {index} in {config.template_csv_path}"
                )
            with rdBase.BlockLogs():
                rxn.Initialize()
            self.reactions.append(rxn)
            self.usable_templates.append(
                rxn.GetNumReactantTemplates() == (2 if count == 2 else 1)
            )

    def reset(self, target_features, start_smiles):
        self.target_features = target_features
        self.start_features = torch.from_numpy(
            np.stack([self.fingerprints.from_smiles(smiles) for smiles in start_smiles])
        ).to(self.device)
        self.routes = [SynthesisRoute(smiles) for smiles in start_smiles]
        self.current_mols = [
            Chem.MolFromSmiles(smiles) if smiles else None for smiles in start_smiles
        ]
        self.pending_templates = [None] * self.batch_size
        self.active = [True] * self.batch_size
        self.seq_types, self.seq_templates, self.seq_fps = [], [], []
        return self._route_batch()

    def _route_batch(self):
        if self.seq_types:
            types = torch.stack(self.seq_types, dim=1)
            templates = torch.stack(self.seq_templates, dim=1)
            fingerprints = torch.stack(self.seq_fps, dim=1)
        else:
            types = torch.empty(
                (self.batch_size, 0), dtype=torch.long, device=self.device
            )
            templates = torch.empty_like(types)
            fingerprints = self.start_features.unsqueeze(1)[:, :0]
        return RouteBatch(
            self.target_features, self.start_features, types, templates, fingerprints
        )

    def _run_reaction(self, template, mol_a, mol_b=None):
        rxn = self.reactions[template]
        try:
            if mol_b is None:
                product_sets = rxn.RunReactants((mol_a,))
            else:
                product_sets = rxn.RunReactants((mol_a, mol_b)) + rxn.RunReactants(
                    (mol_b, mol_a)
                )
        except (ValueError, RuntimeError):
            return None

        with rdBase.BlockLogs():
            products = [
                product
                for group in product_sets
                for product in group
                if Chem.SanitizeMol(product, catchErrors=True)
                == Chem.SanitizeFlags.SANITIZE_NONE
            ]
        if len(products) <= 1:
            return products[0] if products else None
        if self.mode == "qed":
            try:
                return max(products, key=QED.default)
            except (ValueError, RuntimeError):
                return products[0]
        return random.choice(products)

    def check_template(self, index, template):
        mol = self.current_mols[index]
        return (
            mol is not None
            and 0 <= template < len(self.reactions)
            and self.usable_templates[template]
            and self.reactions[template].IsMoleculeReactant(mol)
        )

    def check_reactant(self, index, smiles):
        if not smiles:
            return False
        mol_a = self.current_mols[index]
        if mol_a is None:
            return True
        mol_b = Chem.MolFromSmiles(smiles)
        template = self.pending_templates[index]
        if mol_b is None or template is None:
            return False
        rxn = self.reactions[template]
        try:
            return bool(
                rxn.RunReactants((mol_a, mol_b)) or rxn.RunReactants((mol_b, mol_a))
            )
        except (ValueError, RuntimeError):
            return False

    def _apply_reaction(self, index, template, reactant="", mol_b=None):
        product = self._run_reaction(template, self.current_mols[index], mol_b)
        if product is None:
            return False
        self.current_mols[index] = product
        self.pending_templates[index] = None
        self.routes[index].steps.append(
            ReactionStep(template, reactant, Chem.MolToSmiles(product))
        )
        return True

    def step(self, actions, templates, smiles, fingerprints):
        types = [-100] * self.batch_size
        used_templates = [0] * self.batch_size
        accepted_reactants = [False] * self.batch_size
        for i, (action, template, reactant) in enumerate(
            zip(actions, templates, smiles)
        ):
            if not self.active[i]:
                continue
            types[i] = action

            if action == 2:
                self.active[i] = False
            elif action == 0:
                if template == -1:
                    self.active[i] = False
                    continue
                used_templates[i] = template
                if self.is_bimol[template]:
                    self.pending_templates[i] = template
                else:
                    self.active[i] = self._apply_reaction(i, template)
            elif action == 1:
                mol_b = Chem.MolFromSmiles(reactant) if reactant else None
                if mol_b is None:
                    self.active[i] = False
                elif self.current_mols[i] is None:
                    self.current_mols[i] = mol_b
                    self.routes[i].start_mol = reactant
                else:
                    self.active[i] = self._apply_reaction(
                        i, self.pending_templates[i], reactant, mol_b
                    )
                accepted_reactants[i] = self.active[i]

        self.seq_types.append(torch.tensor(types, dtype=torch.long, device=self.device))
        self.seq_templates.append(
            torch.tensor(used_templates, dtype=torch.long, device=self.device)
        )
        accepted = torch.tensor(
            accepted_reactants, dtype=torch.bool, device=self.device
        ).unsqueeze(1)
        self.seq_fps.append(fingerprints.masked_fill(~accepted, 0.0))
        return self._route_batch()
