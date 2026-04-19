import os
import sys
import subprocess
import argparse
import time

def get_script_path(script_name):
    """Dynamically finds the correct path to the framework scripts."""
    # Location of deploy.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path 1: Same directory (if running from inside scripts/)
    path1 = os.path.join(current_dir, script_name)
    # Path 2: scripts/ subdirectory (if running from Project Root)
    path2 = os.path.join(current_dir, "scripts", script_name)
    
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        print(f"[ERROR] Could not find {script_name} in {current_dir} or {current_dir}/scripts/")
        sys.exit(1)

def run_stage(command, stage_name):
    """Executes a framework script and monitors for success."""
    print(f"\n{'='*70}")
    print(f" >> STAGE: {stage_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    try:
        subprocess.run(command, check=True)
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {stage_name} finished in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {stage_name} failed with exit code {e.returncode}")
        print("Aborting pipeline.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Vitis AI Master Deployment Pipeline")
    
    parser.add_argument('--model', type=str, required=True, help='Model ID')
    parser.add_argument('--dataset', type=str, help='Dataset ID')
    parser.add_argument('--prune', type=float, help='Enable Pruning ratio')
    parser.add_argument('--fast_ft', action='store_true', help='Enable Fast Fine-Tuning')
    parser.add_argument('--subset', type=int, default=200, help='Calib subset length')
    parser.add_argument('--skip_inspect', action='store_true', help='Skip Inspection')
    
    args = parser.parse_args()
    pipeline_start = time.time()
    model_id = args.model
    dataset_arg = ["--dataset", args.dataset] if args.dataset else []

    # --- 1. INSPECTION ---
    if not args.skip_inspect:
        target = get_script_path("run_inspector.py")
        cmd = [sys.executable, target, "--model", model_id] + dataset_arg
        if not run_stage(cmd, "Inspection"): return

    # --- 2. OPTIMIZATION (PRUNING) ---
    if args.prune:
        target = get_script_path("run_optimizer.py")
        cmd = [sys.executable, target, "--model", model_id, "--ratio", str(args.prune)]
        if not run_stage(cmd, "Optimizer (Pruning)"): return

    # --- 3. QUANTIZATION (CALIBRATION) ---
    target = get_script_path("run_quantizer.py")
    cmd = [
        sys.executable, target, 
        "--model", model_id, 
        "--quant_mode", "calib", 
        "--subset_len", str(args.subset)
    ] + dataset_arg
    if args.fast_ft: cmd.append("--fast_ft")
    if not run_stage(cmd, "Quantization: Phase 1 (Calibration)"): return

    # --- 4. QUANTIZATION (TEST/EXPORT) ---
    target = get_script_path("run_quantizer.py") # Same script, different mode
    cmd = [
        sys.executable, target, 
        "--model", model_id, 
        "--quant_mode", "test"
    ] + dataset_arg
    if args.fast_ft: cmd.append("--fast_ft")
    if not run_stage(cmd, "Quantization: Phase 2 (Export)"): return

    # --- 5. COMPILATION ---
    target = get_script_path("run_compiler.py")
    cmd = [sys.executable, target, "--model", model_id]
    if not run_stage(cmd, "Compilation"): return

    # --- FINAL SUMMARY ---
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'#'*70}\n  PIPELINE COMPLETE in {total_elapsed/60:.2f}m\n{'#'*70}")

if __name__ == "__main__":
    main()