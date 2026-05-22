"""
Vitis AI compilation stage.

Wraps the `vai_c_xir` command-line compiler. Reads the quantized xmodel
produced by run_quantizer.py and emits a DPU-ready xmodel targeting the
architecture defined in board_config.DPU_ARCH_PATH.
"""
import os
import sys
import subprocess
import argparse

# Project-root import path (PROJECT_ROOT + scripts/common/ added to sys.path).
from _bootstrap import PROJECT_ROOT  # noqa: F401

from model_config import get_active_model
from board_config import DPU_ARCH_PATH


def run_compiler(model_id):
    """Compile the quantized xmodel for the configured DPU architecture."""
    m_cfg = get_active_model(model_id)
    # Build directory keeps the human-readable name; xmodel filenames use
    # the canonical model_id so board runners can locate them.
    build_dir_name = m_cfg['name'].lower()

    # Quantized xmodel is named as <model_id>_int.xmodel (set by run_quantizer.py).
    quant_dir = os.path.join("build", build_dir_name, "quantize_result")
    quant_model = os.path.join(quant_dir, f"{model_id}_int.xmodel")

    if not os.path.exists(quant_model):
        print(f"[ERROR] Quantized model not found: {quant_model}")
        print(f"[HINT]  Ensure run_quantizer.py finished successfully in 'test' mode.")
        sys.exit(1)

    output_dir = os.path.join("build", build_dir_name, "compiled")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Compilation: {m_cfg['name']} ===")
    print(f"=== Target Architecture: {DPU_ARCH_PATH} ===")

    command = [
        "vai_c_xir",
        "--xmodel", quant_model,
        "--arch", DPU_ARCH_PATH,
        "--net_name", f"{model_id}_kria",
        "--output_dir", output_dir,
    ]
    print(f"Executing: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"=== Compilation Successful ===")
        print(f"Final Model: {output_dir}/{model_id}_kria.xmodel")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Compilation failed:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model ID from model_config.py. Falls back to '
                             'ACTIVE_MODEL_ID when omitted.')
    args = parser.parse_args()

    run_compiler(args.model)
