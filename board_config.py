"""
Hardware constants and live-power query for the Kria target.

The values below describe the specific KV260 build used in this project:

  - DPU_FINGERPRINT : Hex ID emitted by `xdputil query` on the board.
                      Required by the Vitis AI Inspector to validate the
                      compiled subgraph against the actual silicon.
  - DPU_ARCH_PATH   : Path to the Vitis AI compiler arch.json inside the
                      docker image. Used by run_compiler.py.
  - DPU_PEAK_GOPS   : Theoretical peak compute, derived from the DPU PE
                      count, frequency and 2 ops per MAC.
  - ACTIVE_THREADS  : Default number of consumer threads on the board.
  - BOARD_IP / USER : SCP target for board transfer. Set to None to disable
                       the automatic transfer step.

`get_power_mw` shells out to `xmutil xlnx_platformstats -p` to read SOM
total power. Returns 0.0 outside of the Kria environment, which lets the
board scripts run on a host machine without raising.
"""
import re
import subprocess


# ----- Identification -----
BOARD_NAME = "Xilinx Kria KV260"
DPU_ARCH = "DPUCZDX8G_ISA1_B4096"
DPU_FREQ_MHZ = 300

# Unique hardware ID of the deployed DPU; obtained via `xdputil query`.
DPU_FINGERPRINT = "0x101000056010407"

# Compiler architecture descriptor inside Vitis AI Docker.
DPU_ARCH_PATH = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"


# ----- Performance -----
# Peak GOP/s = DPU_PEs * frequency * 2 ops/MAC, expressed in GOP/s.
DPU_PEAK_GOPS = (4096 * DPU_FREQ_MHZ * 2) / 1000

# KV260 supports up to 4 concurrent DPU runners.
# Validate that ACTIVE_THREADS is within hardware limits.
ACTIVE_THREADS = 2
if not (1 <= ACTIVE_THREADS <= 4):
    raise ValueError(f"ACTIVE_THREADS must be between 1 and 4 for KV260. Got: {ACTIVE_THREADS}")

def get_power_mw():
    """
    Read the current SOM total power in milliwatts.

    Returns 0.0 when called outside of the Kria environment so that
    PowerMonitor degrades gracefully on a developer machine.
    """
    try:
        output = subprocess.check_output(
            ["xmutil", "xlnx_platformstats", "-p"],
            stderr=subprocess.STDOUT,
            encoding='utf-8',
        )
        match = re.search(r"SOM total power\s+:\s+(\d+)\s+mW", output)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception:
        return 0.0
