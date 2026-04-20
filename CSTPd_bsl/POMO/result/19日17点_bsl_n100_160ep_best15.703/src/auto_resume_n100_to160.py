##########################################################################################
# Auto-resume helper for the current n100 baseline run.
#
# This script intentionally keeps the normal trainer untouched. It waits for an existing
# training process to finish, finds the latest numeric checkpoint in that run folder, and
# continues training to a larger epoch budget in a new result folder.

import argparse
import glob
import json
import os
import re
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


import train_n100 as config  # noqa: E402

from utils.utils import copy_all_src, create_logger  # noqa: E402


def _process_alive(pid):
    if pid is None:
        return False
    return os.path.exists("/proc/{}".format(pid))


def _wait_for_process(pid, poll_sec):
    if pid is None:
        return

    print("Waiting for training process PID {} to finish...".format(pid), flush=True)
    while _process_alive(pid):
        time.sleep(poll_sec)
    print("Training process PID {} finished.".format(pid), flush=True)


def _latest_numeric_checkpoint(run_dir):
    latest_epoch = None
    pattern = os.path.join(run_dir, "checkpoint-*.pt")

    for checkpoint_path in glob.glob(pattern):
        match = re.search(r"checkpoint-(\d+)\.pt$", os.path.basename(checkpoint_path))
        if match is None:
            continue
        epoch = int(match.group(1))
        latest_epoch = epoch if latest_epoch is None else max(latest_epoch, epoch)

    if latest_epoch is None:
        raise FileNotFoundError("No numeric checkpoint found in {}".format(run_dir))

    return latest_epoch


def _read_elapsed_time(run_dir):
    progress_path = os.path.join(run_dir, "training_progress.json")
    if not os.path.exists(progress_path):
        return 0.0

    with open(progress_path, encoding="utf-8") as f:
        progress = json.load(f)

    return float(progress.get("total_training_time_sec") or 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-epochs", type=int, default=160)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    _wait_for_process(args.wait_pid, args.poll_sec)

    checkpoint_epoch = _latest_numeric_checkpoint(source_dir)
    if checkpoint_epoch >= args.target_epochs:
        print(
            "Latest checkpoint epoch {} already reaches target {}. Nothing to resume.".format(
                checkpoint_epoch,
                args.target_epochs,
            ),
            flush=True,
        )
        return

    config.trainer_params["epochs"] = args.target_epochs
    config.trainer_params["model_load"] = {
        "enable": True,
        "path": source_dir,
        "epoch": checkpoint_epoch,
    }
    config.trainer_params["training_elapsed_time_sec"] = _read_elapsed_time(source_dir)
    config.logger_params["log_file"]["desc"] = (
        "train__baseline_ctspd_n100__resume_e{}_to{}_bs{}".format(
            checkpoint_epoch,
            args.target_epochs,
            config.trainer_params["train_batch_size"],
        )
    )

    print(
        "Resuming from {} checkpoint-{}.pt to epoch {}.".format(
            source_dir,
            checkpoint_epoch,
            args.target_epochs,
        ),
        flush=True,
    )

    create_logger(**config.logger_params)
    config._print_config()

    trainer = config.Trainer(
        env_params=config.env_params,
        model_params=config.model_params,
        optimizer_params=config.optimizer_params,
        trainer_params=config.trainer_params,
    )

    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    main()
