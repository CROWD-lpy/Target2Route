import pandas as pd


def load_templates(config):
    templates = pd.read_csv(config.template_csv_path)
    if len(templates) != config.num_templates:
        raise ValueError(
            f"{config.template_csv_path} has {len(templates)} templates; expected {config.num_templates}"
        )
    required = {"forward_template", "num_reactants"}
    if not required.issubset(templates):
        raise ValueError(f"{config.template_csv_path} must include {sorted(required)}")
    if "Trace_Template_ID" in templates and templates[
        "Trace_Template_ID"
    ].tolist() != list(range(len(templates))):
        raise ValueError(
            f"Trace_Template_ID must follow the original zero-based row order in {config.template_csv_path}"
        )
    return templates
