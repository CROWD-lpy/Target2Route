import csv
from dataclasses import dataclass, field
from pathlib import Path

MIN_REACTIONS = 3
MAX_REACTIONS = 6
SUCCESSFUL_ENDS = {"normal_end", "truncated_end"}
VALID_OUTPUT_FATES = SUCCESSFUL_ENDS | {"threshold_fail"}


@dataclass
class ReactionStep:
    template_id: int
    reactant: str
    product: str


@dataclass
class SynthesisRoute:
    start_mol: str
    steps: list[ReactionStep] = field(default_factory=list)

    @property
    def product(self):
        return self.steps[-1].product if self.steps else self.start_mol


ROUTE_COLUMNS = [
    "Episode",
    "Mode",
    "Target_Idx",
    "Target_Seq",
    "Final_Product",
    "Fate",
    "Termination",
    "QED_Score",
    "Vina_Score",
    "SaScore",
    "Reward",
    "start_mol",
    "Start_Reactant",
    "Rxn_Count",
] + [
    f"Rxn{step}_{field}"
    for step in range(1, MAX_REACTIONS + 1)
    for field in ("Template", "Reactant", "Product")
]


def route_record(episode, mode, target, route, termination, scores=None, index=None):
    record = {
        "Episode": episode,
        "Mode": mode,
        "Target_Idx": target.id,
        "Target_Seq": target.sequence,
        "Final_Product": route.product,
        "Fate": termination,
        "Termination": termination,
        "start_mol": route.start_mol,
        "Start_Reactant": route.start_mol,
        "Rxn_Count": len(route.steps),
    }
    if scores is not None:
        record.update(
            Fate=scores.fates[index],
            Reward=scores.rewards[index].item(),
            QED_Score=0.0,
            Vina_Score=0.0,
            SaScore=0.0,
        )
        if scores.qed_scores[index] > 0 and scores.vina_scores[index] < 0:
            record.update(
                QED_Score=scores.qed_scores[index].item(),
                Vina_Score=scores.vina_scores[index].item(),
                SaScore=scores.sa_scores[index].item(),
            )
    for i, step in enumerate(route.steps, start=1):
        record.update(
            {
                f"Rxn{i}_Template": step.template_id,
                f"Rxn{i}_Reactant": step.reactant or "N/A",
                f"Rxn{i}_Product": step.product,
            }
        )
    return record


def write_routes(path, rows=()):
    path = Path(path)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
