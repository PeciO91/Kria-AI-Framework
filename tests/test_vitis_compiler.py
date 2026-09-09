import unittest
from pathlib import Path

from kria_ai.common.vitis.compiler import build_compile_command


class TestCompilerCommand(unittest.TestCase):
    def test_builds_explicit_command(self):
        command = build_compile_command(
            input_xmodel=Path("build/model/quantize_result/model_int.xmodel"),
            arch_path=Path("/opt/vitis/arch.json"),
            output_dir=Path("build/model/compiled"),
            net_name="model_kria",
        )
        self.assertEqual(command[0], "vai_c_xir")
        self.assertEqual(command[command.index("--net_name") + 1], "model_kria")
        self.assertEqual(command[command.index("--arch") + 1], "/opt/vitis/arch.json")


if __name__ == "__main__":
    unittest.main()
