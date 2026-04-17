import os
import sys
import subprocess

# --- Path auto-fix to find configs in project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from board_config import DPU_ARCH_PATH

def run_compiler():
    # 1. Load configuration
    m_cfg = get_active_model()
    model_name = m_cfg['name'].lower()
    
    # 2. Define input/output paths
    # Input is the xmodel from the quantizer output folder
    quant_dir = os.path.join("build", model_name, "quantize_result")
    
    # Automatically find the generated xmodel (usually named after the class, e.g., ResNet_int.xmodel)
    quant_model = None
    if os.path.exists(quant_dir):
        for f in os.listdir(quant_dir):
            if f.endswith(".xmodel") and not f.startswith("deploy"):
                quant_model = os.path.join(quant_dir, f)
                break
            
    if not quant_model:
        print(f"Error: No quantized xmodel found in {quant_dir}. Did you run the quantizer in 'test' mode?")
        return

    # Output directory for the compiled instructions
    output_dir = os.path.join("build", model_name, "compiled")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Compilation: {m_cfg['name']} ===")
    print(f"=== Target Architecture: {DPU_ARCH_PATH} ===")

    # 3. Construct the Vitis AI Compiler command (vai_c_xir)
    # --xmodel:   quantized model from NNDCT
    # --arch:     architecture description from board_config
    # --net_name: name for the final hardware-ready xmodel
    # --output_dir: where the compiled instructions will be stored
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
        # We capture output to show it in the console
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"=== Compilation Successful! ===")
        print(f"Final Model: {output_dir}/{model_name}_kria.xmodel")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Compilation failed!\n{e.stderr}")

if __name__ == "__main__":
    run_compiler()