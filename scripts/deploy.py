import os
import sys
import subprocess
import argparse
import time

# --- Path auto-fix ---
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_ROOT, '..'))

if SCRIPT_ROOT not in sys.path: sys.path.insert(0, SCRIPT_ROOT)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from model_config import get_active_model
try:
    from board_config import BOARD_IP, BOARD_USER
except ImportError:
    BOARD_IP = None
    BOARD_USER = "root"

def get_script_path(script_name):
    path1 = os.path.join(SCRIPT_ROOT, script_name)
    path2 = os.path.join(PROJECT_ROOT, "scripts", script_name)
    if os.path.exists(path1): return path1
    if os.path.exists(path2): return path2
    print(f"[ERROR] Could not find {script_name}"); sys.exit(1)

def run_stage(command, stage_name):
    print(f"\n{'='*70}\n >> STAGE: {stage_name}\n{'='*70}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] {stage_name} failed. Aborting.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Vitis AI Master Deployment Pipeline")
    parser.add_argument('--model', type=str, required=True, help='Model ID')
    parser.add_argument('--dataset', type=str, help='Dataset ID')
    parser.add_argument('--prune', type=float, help='Pruning ratio')
    parser.add_argument('--fast_ft', action='store_true', help='Enable Fast Fine-Tuning')
    parser.add_argument('--subset', type=int, default=200, help='Calib subset length')
    parser.add_argument('--skip_inspect', action='store_true', help='Skip Inspection')
    parser.add_argument('--ip', type=str, default=BOARD_IP, help='Kria IP')
    parser.add_argument('--user', type=str, default=BOARD_USER, help='SSH User')
    
    args = parser.parse_args()
    pipeline_start = time.time()
    
    m_cfg = get_active_model(args.model)
    folder_name = m_cfg['name'].lower()
    
    dataset_arg = ["--dataset", args.dataset] if args.dataset else []
    prune_arg = ["--prune_threshold", str(args.prune)] if args.prune else []

    # 1. INSPECTION
    if not args.skip_inspect:
        if not run_stage([sys.executable, get_script_path("run_inspector.py"), "--model", args.model] + dataset_arg + prune_arg, "Inspection"): return

    # 2. OPTIMIZATION
    if args.prune:
        if not run_stage([sys.executable, get_script_path("run_optimizer.py"), "--model", args.model, "--ratio", str(args.prune)], "Optimizer"): return

    # 3. QUANTIZATION (CALIB)
    target_q = get_script_path("run_quantizer.py")
    cmd_q = [sys.executable, target_q, "--model", args.model, "--quant_mode", "calib", "--subset_len", str(args.subset)] + dataset_arg + prune_arg
    if args.fast_ft: cmd_q.append("--fast_ft")
    if not run_stage(cmd_q, "Quantization: Phase 1"): return

    # 4. QUANTIZATION (TEST)
    cmd_q[cmd_q.index("calib")] = "test"
    if not run_stage(cmd_q, "Quantization: Phase 2"): return

    # 5. COMPILATION
    # Uses corrected run_compiler.py that accepts model argument
    if not run_stage([sys.executable, get_script_path("run_compiler.py"), "--model", args.model], "Compilation"): return

# --- 6. AUTOMATED TRANSFER ---
    if args.ip:
        local_model = os.path.join(PROJECT_ROOT, "build", folder_name, "compiled", f"{folder_name}_kria.xmodel")
        remote_dest = f"{args.user}@{args.ip}:/home/{args.user}/"
        
        print(f"\n{'='*70}\n >> STAGE: Transfer to Kria (Full Package)\n{'='*70}")
        
        # List of essential files to run inference on the board
        files_to_send = [
            local_model,
            get_script_path("run_inference.py"),
            get_script_path("run_detection.py"),
            get_script_path("run_segmentation.py"),
            get_script_path("detection_utils.py"),
            get_script_path("board_utils.py"),
            get_script_path("model_config.py"),
            get_script_path("dataset_config.py"),
            get_script_path("board_config.py")
        ]

        # Filter out missing files to avoid SCP errors
        existing_files = [f for f in files_to_send if os.path.exists(f)]
        
        # -o BatchMode=yes: Fails immediately if password is required (prevents hanging)
        transfer_cmd = [
            "scp", 
            "-o", "StrictHostKeyChecking=no", 
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes" 
        ] + existing_files + [remote_dest]
        
        try:
            subprocess.run(transfer_cmd, check=True)
            print(f"\n[SUCCESS] Model and scripts transferred to {args.ip}")
            print(f"[TIP] To run: ssh {args.user}@{args.ip} 'python3 run_inference.py --model {args.model}'")
        except subprocess.CalledProcessError:
            print(f"\n[ERROR] Transfer failed. Verify SSH keys or IP address.")

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'#'*70}\n  PIPELINE COMPLETE in {total_elapsed/60:.2f}m\n{'#'*70}")

if __name__ == "__main__":
    main()