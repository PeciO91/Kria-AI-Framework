import os
import sys
import subprocess
import argparse

# --- Path auto-fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from board_config import DPU_ARCH_PATH

def run_compiler(model_id):
    # 1. Load configuration (FIXED: Now uses the passed model_id)
    m_cfg = get_active_model(model_id)
    model_name = m_cfg['name'].lower()
    
    # 2. Define input/output paths
    quant_dir = os.path.join("build", model_name, "quantize_result")
    
    quant_model = None
    if os.path.exists(quant_dir):
        for f in os.listdir(quant_dir):
            if f.endswith(".xmodel") and not f.startswith("deploy"):
                quant_model = os.path.join(quant_dir, f)
                break
            
    if not quant_model:
        print(f"Error: No quantized xmodel found in {quant_dir}.")
        return

    output_dir = os.path.join("build", model_name, "compiled")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Compilation: {m_cfg['name']} ===")
    print(f"=== Target Architecture: {DPU_ARCH_PATH} ===")

    # 3. Construct the Vitis AI Compiler command
    command = [
        "vai_c_xir",
        "--xmodel", quant_model,
        "--arch", DPU_ARCH_PATH,
        "--net_name", f"{model_name}_kria",
        "--output_dir", output_dir
    ]

    print(f"Executing: {' '.join(command)}")

    # 4. Run the compilation process
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"=== Compilation Successful! ===")
        print(f"Final Model: {output_dir}/{model_name}_kria.xmodel")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Compilation failed!\n{e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model ID from model_config.py')
    args = parser.parse_args()

    # Pass the CLI argument INTO the function
    run_compiler(args.model)