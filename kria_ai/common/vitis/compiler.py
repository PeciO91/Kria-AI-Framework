import subprocess
from pathlib import Path


def build_compile_command(*, input_xmodel, arch_path, output_dir, net_name):
    return [
        "vai_c_xir",
        "--xmodel",
        str(input_xmodel),
        "--arch",
        str(arch_path),
        "--net_name",
        net_name,
        "--output_dir",
        str(output_dir),
    ]


def compile_xmodel(*, input_xmodel, arch_path, output_dir, net_name):
    input_xmodel = Path(input_xmodel)
    arch_path = Path(arch_path)
    output_dir = Path(output_dir)
    if not input_xmodel.is_file():
        raise FileNotFoundError(f"Quantized XMODEL not found: {input_xmodel}")
    if not arch_path.is_file():
        raise FileNotFoundError(f"DPU architecture descriptor not found: {arch_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_compile_command(
        input_xmodel=input_xmodel,
        arch_path=arch_path,
        output_dir=output_dir,
        net_name=net_name,
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output_path = output_dir / f"{net_name}.xmodel"
    if not output_path.is_file():
        raise RuntimeError(
            f"Vitis compiler completed without expected output {output_path}. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return output_path
