import subprocess
import re

# Hardware identification for report generation
BOARD_NAME = "Xilinx Kria KV260"
DPU_ARCH = "DPUCZDX8G_ISA1_B4096"
DPU_FREQ_MHZ = 300

# The DPU Fingerprint is the unique hardware ID of your DPU configuration.
# Obtained via 'xdputil query' on the target board.
DPU_FINGERPRINT = "0x101000056010407"

# Theoretical Peak Performance Calculation (GOP/s)
# Formula: (4096 ops * 300 MHz * 2 operations per MAC) / 1000
DPU_PEAK_GOPS = (4096 * DPU_FREQ_MHZ * 2) / 1000 

def get_power_mw():
    """
    Reads the current SOM power consumption from the Kria system.
    Returns value in milliwatts (mW).
    """
    try:
        # Calls the platformstats utility to get power metrics
        output = subprocess.check_output(
            ["xmutil", "xlnx_platformstats", "-p"], 
            stderr=subprocess.STDOUT, 
            encoding='utf-8'
        )
        # Parses the output to find total SOM power
        match = re.search(r"SOM total power\s+:\s+(\d+)\s+mW", output)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        # Returns 0.0 if not running on Kria or command fails
        return 0.0