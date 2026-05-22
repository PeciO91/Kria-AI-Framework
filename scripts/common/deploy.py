"""
End-to-end deployment driver.

Chains the four host-side stages (Inspector, optional Optimizer, Quantizer
calib + test, Compiler) and then transfers the compiled xmodel along with
all board-side scripts to the Kria target. Transfer is configurable via
``--transfer {scp,rsync,local,none}``; SCP remains the default.

Each stage is a separate subprocess so that quantizer / compiler errors do
not interrupt the next deployment in a sweep.
"""
import os
import sys
import shutil
import subprocess
import argparse
import time

# Project-root import path (PROJECT_ROOT + scripts/common/ added to sys.path).
from _bootstrap import PROJECT_ROOT, SCRIPTS_ROOT, COMMON_DIR

from model_config import get_active_model, ACTIVE_MODEL_ID
try:
    from board_config import BOARD_IP, BOARD_USER
except ImportError:
    BOARD_IP = None
    BOARD_USER = "root"


# Task-type -> sub-folder under scripts/ that holds the on-board runner +
# any task-only helpers. Host-side stages (inspector, quantizer, optimizer,
# compiler) always live in scripts/common/ regardless of task.
TASK_DIRS = {
    "classification": os.path.join(SCRIPTS_ROOT, "classification"),
    "detection":      os.path.join(SCRIPTS_ROOT, "detection"),
    "segmentation":   os.path.join(SCRIPTS_ROOT, "segmentation"),
}


def get_script_path(script_name, task=None):
    """Resolve a stage / runner script across the split layout.

    Search order:
      1. ``scripts/common/`` (task-agnostic stages & utilities)
      2. ``scripts/<task>/`` if ``task`` is given (task-specific runner)
      3. every ``scripts/<task>/`` folder as a last-resort fallback
    """
    candidates = [os.path.join(COMMON_DIR, script_name)]
    if task and task in TASK_DIRS:
        candidates.append(os.path.join(TASK_DIRS[task], script_name))
    candidates.extend(os.path.join(d, script_name) for d in TASK_DIRS.values())
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    print(f"[ERROR] Could not find {script_name}")
    sys.exit(1)


# Per-task board-side payloads. ``board_utils.py`` is always included by the
# common-folder payload below; these lists hold only the task-specific files.
TASK_BOARD_FILES = {
    "classification": ["run_inference.py"],
    "detection":      ["run_detection.py", "detection_utils.py"],
    "segmentation":   ["run_segmentation.py"],
}


def get_project_file(file_name):
    candidate = os.path.join(PROJECT_ROOT, file_name)
    if os.path.exists(candidate):
        return candidate
    print(f"[ERROR] Could not find {file_name}")
    sys.exit(1)


def run_stage(command, stage_name):
    """Run a stage subprocess and return True on success."""
    print(f"\n{'='*70}\n >> STAGE: {stage_name}\n{'='*70}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] {stage_name} failed. Aborting.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Vitis AI Master Deployment Pipeline")
    parser.add_argument('--model', type=str,
                        help='Model ID. Falls back to ACTIVE_MODEL_ID '
                             'in model_config.py when omitted.')
    parser.add_argument('--dataset', type=str, help='Dataset ID')
    parser.add_argument('--prune', type=float, help='Pruning ratio')
    parser.add_argument('--method', choices=['iterative', 'onestep'], default='iterative',
                        help='Pruning algorithm (passed to run_optimizer.py). '
                             'iterative=sensitivity analysis (default); '
                             'onestep=EagleEye subnet search.')
    parser.add_argument('--num_subnet', type=int, default=200,
                        help='Subnet candidates for --method onestep (ignored otherwise)')
    parser.add_argument('--ft_epochs', type=int, default=5, help='Fine-tuning epochs for optimizer')
    parser.add_argument('--fast_ft', action='store_true', help='Enable AdaQuant Fast Fine-Tuning')
    parser.add_argument('--subset', type=int, default=200, help='Calibration subset length')
    parser.add_argument('--skip_inspect', action='store_true', help='Skip the inspection stage')
    parser.add_argument('--ip', type=str, default=BOARD_IP, help='Kria board IP')
    parser.add_argument('--user', type=str, default=BOARD_USER, help='SSH user on the board')
    parser.add_argument('--transfer', choices=['scp', 'rsync', 'local', 'none'],
                        default='scp',
                        help='Transfer method after compilation: '
                             'scp=copy to Kria via SCP (default); '
                             'rsync=copy to Kria via rsync (faster on re-deploys); '
                             'local=copy to a local directory (--local_dest); '
                             'none=skip transfer entirely.')
    parser.add_argument('--local_dest', type=str,
                        help='Destination directory for --transfer local '
                             '(e.g. an SD-card mount point).')

    args = parser.parse_args()
    pipeline_start = time.time()

    # Resolve --model fallback up-front so we can forward a concrete ID to
    # every stage subprocess (avoids passing the literal string "None").
    if not args.model:
        args.model = ACTIVE_MODEL_ID
        print(f"[INFO] No --model provided, using ACTIVE_MODEL_ID='{args.model}'.")
    m_cfg = get_active_model(args.model)
    # Build directory uses human-readable name; xmodel filename uses
    # the canonical model_id so it matches what board runners look for.
    build_dir_name = m_cfg['name'].lower()

    dataset_arg = ["--dataset", args.dataset] if args.dataset else []

    # 1. Inspection
    if not args.skip_inspect:
        if not run_stage(
            [sys.executable, get_script_path("run_inspector.py"), "--model", args.model]
            + dataset_arg, "Inspection"):
            return

    # 2. Optimization (pruning) - optional
    if args.prune:
        opt_extra = ["--method", args.method, "--subset_len", str(args.subset)]
        if args.method == 'onestep':
            opt_extra += ["--num_subnet", str(args.num_subnet)]
        if not run_stage(
            [sys.executable, get_script_path("run_optimizer.py"), "--model", args.model,
             "--ratio", str(args.prune), "--epochs", str(args.ft_epochs)]
            + dataset_arg + opt_extra, "Optimizer"):
            return

    # 3. Quantization (calibration phase)
    target_q = get_script_path("run_quantizer.py")
    cmd_q = [sys.executable, target_q, "--model", args.model,
             "--quant_mode", "calib", "--subset_len", str(args.subset)] + dataset_arg
    if args.fast_ft:
        cmd_q.append("--fast_ft")
    if not run_stage(cmd_q, "Quantization: Phase 1"):
        return

    # 4. Quantization (export phase)
    cmd_q[cmd_q.index("calib")] = "test"
    if not run_stage(cmd_q, "Quantization: Phase 2"):
        return

    # 5. Compilation
    if not run_stage(
        [sys.executable, get_script_path("run_compiler.py"), "--model", args.model],
        "Compilation"):
        return

    # 6. Transfer to Kria board (xmodel + only the runner files relevant to
    # this task). board_utils.py is shared by every runner; the per-task
    # files come from TASK_BOARD_FILES.
    task_type = m_cfg.get("type", "classification")
    task_files = TASK_BOARD_FILES.get(task_type, TASK_BOARD_FILES["classification"])

    transfer_payload = [
        os.path.join(PROJECT_ROOT, "build", build_dir_name, "compiled",
                     f"{args.model}_kria.xmodel"),
        get_script_path("board_utils.py"),
        get_project_file("model_config.py"),
        get_project_file("dataset_config.py"),
        get_project_file("board_config.py"),
    ]
    for name in task_files:
        transfer_payload.append(get_script_path(name, task=task_type))
    existing_files = [f for f in transfer_payload if os.path.exists(f)]

    # Pick the runner filename that the user should invoke on the board.
    task_runner_map = {
        "classification": "run_inference.py",
        "detection":      "run_detection.py",
        "segmentation":   "run_segmentation.py",
    }
    runner = task_runner_map.get(task_type, "run_inference.py")
    dataset_tip = f" --dataset {args.dataset}" if args.dataset else ""

    if args.transfer == 'none':
        print(f"\n{'='*70}\n >> STAGE: Transfer skipped (--transfer none)\n{'='*70}")
        print(f"[INFO] Compiled xmodel: "
              f"build/{build_dir_name}/compiled/{args.model}_kria.xmodel")

    elif args.transfer == 'local':
        if not args.local_dest:
            print(f"\n[ERROR] --transfer local requires --local_dest <directory>.")
        else:
            print(f"\n{'='*70}\n >> STAGE: Transfer to local "
                  f"({args.local_dest})\n{'='*70}")
            try:
                os.makedirs(args.local_dest, exist_ok=True)
                for src in existing_files:
                    shutil.copy2(src, os.path.join(args.local_dest, os.path.basename(src)))
                print(f"\n[SUCCESS] Files copied to {args.local_dest}")
            except OSError as e:
                print(f"\n[ERROR] Local copy failed: {e}")

    elif args.transfer in ('scp', 'rsync'):
        if not args.ip:
            print(f"\n[WARN] --transfer {args.transfer} but no --ip / BOARD_IP set. "
                  f"Skipping transfer.")
        else:
            remote_dest = f"{args.user}@{args.ip}:/home/{args.user}/"
            print(f"\n{'='*70}\n >> STAGE: Transfer to Kria via "
                  f"{args.transfer.upper()}\n{'='*70}")

            ssh_opts = [
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes",
            ]
            if args.transfer == 'scp':
                # BatchMode=yes prevents SSH from hanging on a missing key.
                transfer_cmd = ["scp"] + ssh_opts + existing_files + [remote_dest]
            else:
                # rsync over SSH; -avz preserves attrs, archives, compresses.
                transfer_cmd = (["rsync", "-avz", "-e", "ssh " + " ".join(ssh_opts)]
                                + existing_files + [remote_dest])

            try:
                subprocess.run(transfer_cmd, check=True)
                print(f"\n[SUCCESS] Model and scripts transferred to {args.ip}")
                print(f"[TIP] Run on board: ssh {args.user}@{args.ip} "
                      f"'python3 {runner} --model {args.model}{dataset_tip}'")
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] Transfer failed. Verify SSH keys or IP address.")
            except FileNotFoundError:
                print(f"\n[ERROR] '{args.transfer}' executable not found on PATH.")

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'#'*70}\n  PIPELINE COMPLETE in {total_elapsed/60:.2f}m\n{'#'*70}")


if __name__ == "__main__":
    main()
