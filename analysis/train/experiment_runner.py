import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PRETRAIN_ARGS = {
    "configuration": "--configuration",
    "dataset_path": "--dataset-path",
    "mlm_probability": "--mlm-probability",
    "span_mask_probability": "--span-mask-probability",
    "num_epochs": "--num-epochs",
    "batch_size": "--batch-size",
    "test_size": "--test-size",
    "output_path": "--output-path",
    "learning_rate": "--learning-rate",
}

FINETUNE_ARGS = {
    "mode": "--mode",
    "dataset_path": "--dataset-path",
    "test_size": "--test-size",
    "num_epochs": "--num-epochs",
    "batch_size": "--batch-size",
    "gradient_accumulation_steps": "--gradient-accumulation-steps",
    "output_path": "--output-path",
    "learning_rate": "--learning-rate",
    "freeze_layers_up_to": "--freeze-layers-up-to",
    "pretrained": "--pretrained",
    "from_init": "--from-init",
    "attn_pool": "--attn-pool",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run pretraining or finetuning from JSON config")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_run_config(config, run):
    merged = dict(config.get("defaults", {}))
    for key, value in config.items():
        if key not in {"defaults", "models", "launcher", "task"}:
            merged[key] = value
    merged["python"] = sys.executable
    merged.update(run)
    return merged


def add_cli_args(command, arg_map, values):
    for key, flag in arg_map.items():
        value = values.get(key)
        if value is None or value is False:
            continue
        if isinstance(value, bool):
            command.append(flag)
        else:
            command.extend([flag, str(value)])


def build_base_command(task, launcher):
    script_name = "train.py" if task == "pretrain" else "finetune.py"
    script_path = ROOT / "analysis" / "train" / script_name
    if launcher.get("type") == "torchrun":
        command = ["torchrun"]
        if "nproc_per_node" in launcher:
            command.extend(["--nproc_per_node", str(launcher["nproc_per_node"])])
        if "master_port" in launcher:
            command.extend(["--master_port", str(launcher["master_port"])])
    else:
        command = [sys.executable]
    command.append(str(script_path))
    return command


def build_command(task, launcher, run_config):
    command = build_base_command(task, launcher)
    arg_map = PRETRAIN_ARGS if task == "pretrain" else FINETUNE_ARGS
    add_cli_args(command, arg_map, run_config)
    return command


def build_custom_command(run_config):
    template = run_config.get("command")
    if not template:
        raise ValueError("Custom run requires 'command'")

    if isinstance(template, str):
        rendered = template.format(**run_config)
        return shlex.split(rendered)

    if isinstance(template, list):
        return [str(part).format(**run_config) for part in template]

    raise ValueError("'command' must be a string or list")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    task = config.get("task")
    if task not in {"pretrain", "finetune"}:
        raise ValueError("Config 'task' must be 'pretrain' or 'finetune'")

    runs = config.get("models", [])
    if not runs:
        raise ValueError("Config must define a non-empty 'models' list")

    launcher = config.get("launcher", {})

    for run in runs:
        run_config = merge_run_config(config, run)
        run_task = run.get("task", task)
        run_name = run.get("name") or run.get("configuration") or run.get("output_path")

        if run_task in {"pretrain", "finetune"}:
            command = build_command(run_task, launcher, run_config)
        elif run_task == "command":
            command = build_custom_command(run_config)
        else:
            raise ValueError(
                f"Unknown run task '{run_task}'. Expected pretrain, finetune, or command."
            )

        print(f"[{run_task}] {run_name}")
        print(" ".join(shlex.quote(part) for part in command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()