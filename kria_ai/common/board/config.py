from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoardConfig:
    id: str
    name: str
    dpu_arch: str
    dpu_frequency_mhz: int
    dpu_fingerprint: str
    compiler_arch_path: Path
    dpu_peak_gops: float
    default_runners: int
    max_runners: int
    power_command: tuple[str, ...]

    def validate_runner_count(self, count):
        if not 1 <= count <= self.max_runners:
            raise ValueError(f"Runner count must be between 1 and {self.max_runners}; got {count}")
        return count


KV260 = BoardConfig(
    id="kv260",
    name="Xilinx Kria KV260",
    dpu_arch="DPUCZDX8G_ISA1_B4096",
    dpu_frequency_mhz=300,
    dpu_fingerprint="0x101000056010407",
    compiler_arch_path=Path("/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"),
    dpu_peak_gops=(4096 * 300 * 2) / 1000,
    default_runners=2,
    max_runners=4,
    power_command=("xmutil", "xlnx_platformstats", "-p"),
)

BOARDS = {KV260.id: KV260}
DEFAULT_BOARD_ID = KV260.id


def get_board(board_id=None):
    resolved_id = board_id or DEFAULT_BOARD_ID
    try:
        return BOARDS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown board {resolved_id!r}; available: {', '.join(BOARDS)}") from error
