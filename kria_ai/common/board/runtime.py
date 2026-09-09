from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    fixed_point: int


@dataclass(frozen=True)
class DpuModel:
    graph: object
    subgraph: object
    inputs: tuple[TensorMetadata, ...]
    outputs: tuple[TensorMetadata, ...]


def _metadata(tensor):
    try:
        name = tensor.name
    except AttributeError:
        name = tensor.get_name()
    try:
        fixed_point = int(tensor.get_attr("fix_point"))
    except Exception as error:
        raise RuntimeError(f"Tensor {name!r} has no fixed-point metadata") from error
    return TensorMetadata(name=name, shape=tuple(tensor.dims), fixed_point=fixed_point)


def load_dpu_model(model_path):
    try:
        import vart
        import xir
    except ImportError as error:
        raise RuntimeError("XIR and VART are required on the target board") from error

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"XMODEL not found: {model_path}")
    graph = xir.Graph.deserialize(str(model_path))
    root = graph.get_root_subgraph()
    children = list(root.toposort_child_subgraph()) if hasattr(root, "toposort_child_subgraph") else list(root.get_children())
    dpu_subgraphs = [child for child in children if child.has_attr("device") and child.get_attr("device").upper() == "DPU"]
    if len(dpu_subgraphs) != 1:
        raise RuntimeError(f"Expected exactly one DPU subgraph in {model_path}; found {len(dpu_subgraphs)}")
    subgraph = dpu_subgraphs[0]
    runner = vart.Runner.create_runner(subgraph, "run")
    try:
        inputs = tuple(_metadata(tensor) for tensor in runner.get_input_tensors())
        outputs = tuple(_metadata(tensor) for tensor in runner.get_output_tensors())
    finally:
        del runner
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one DPU input tensor; found {len(inputs)}")
    return DpuModel(graph=graph, subgraph=subgraph, inputs=inputs, outputs=outputs)


def create_runner(dpu_model):
    try:
        import vart
    except ImportError as error:
        raise RuntimeError("VART is required on the target board") from error
    return vart.Runner.create_runner(dpu_model.subgraph, "run")


def allocate_output_buffers(runner):
    import numpy as np

    return [np.empty(tuple(tensor.dims), dtype=np.int8) for tensor in runner.get_output_tensors()]


def output_dequantization_scales(metadata):
    return tuple(2.0 ** -tensor.fixed_point for tensor in metadata)
