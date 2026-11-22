"""Separate directory production from a training experiment's selected sources."""
from copy import deepcopy
from pathlib import Path
import json

import yaml

from .core import load_config


def make_workflow_configs(config, plan, job_dir):
    if config["data"].get("layout") != "by_source":
        raise ValueError("Workflow selection requires data.layout=by_source")
    if not isinstance(plan, dict) or set(plan) != {"directory_sources", "training_sources", "experiment"}:
        raise ValueError("Workflow requires directory_sources, training_sources and experiment")
    available = config["data"]["sources"]
    for key in ("directory_sources", "training_sources"):
        sources = plan[key]
        if (not isinstance(sources, list) or not sources or any(not isinstance(s, str) for s in sources)
                or len(set(sources)) != len(sources) or not set(sources).issubset(available)):
            raise ValueError(f"{key} must be a nonempty unique subset of configured data.sources")
    name = plan["experiment"]
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("experiment must be a single directory name")
    build, train = deepcopy(config), deepcopy(config)
    build["data"]["sources"] = list(plan["directory_sources"])
    train["data"]["sources"] = list(plan["training_sources"])
    # Dataset selection gets its own checkpoint/metrics namespace; prepared data
    # and HTTP identity remain shared. Adding M is a new experiment, not resume.
    run_root = Path(config["data"]["prepared_root"]).parent
    train["training"]["output_dir"] = str(run_root / "experiments" / name / "train")
    marker = Path(train["training"]["output_dir"]) / "run.json"
    if marker.exists():
        previous = json.loads(marker.read_text())["config"]
        for section in ("data", "models", "service", "writer", "reader", "retrieval", "reward", "training"):
            if previous.get(section) != train.get(section):
                raise ValueError(f"Existing experiment {name} differs in {section}; choose a new experiment name")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (("build_config.yaml", build), ("train_config.yaml", train), ("workflow.yaml", plan)):
        path = job_dir / filename
        with path.open("x") as stream:
            yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
    return build, train


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("plan")
    parser.add_argument("job_dir")
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text())
    build, train = make_workflow_configs(load_config(args.config), plan, args.job_dir)
    print(f"Directory sources: {build['data']['sources']}")
    print(f"Training/validation sources: {train['data']['sources']}")
    print(f"Training output: {train['training']['output_dir']}")
