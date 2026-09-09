from kria_ai.common.board.config import KV260
from kria_ai.common.board.power import read_power_mw


BOARD_NAME = KV260.name
DPU_ARCH = KV260.dpu_arch
DPU_FREQ_MHZ = KV260.dpu_frequency_mhz
DPU_FINGERPRINT = KV260.dpu_fingerprint
DPU_ARCH_PATH = str(KV260.compiler_arch_path)
DPU_PEAK_GOPS = KV260.dpu_peak_gops
ACTIVE_THREADS = KV260.default_runners


def get_power_mw():
    return read_power_mw(KV260.power_command)
