import argparse
from collections.abc import Sequence

from kria_ai.classification.config import get_model
from kria_ai.classification.models import build_model
from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.vitis.inspector import inspect_model


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification inspect")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--board")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    model_config = get_model(args.model)
    board_config = get_board(args.board)
    device = _device(args.device)
    model, _ = build_model(model_config, device=device, checkpoint_path=args.checkpoint)
    height, width = model_config.input_size
    example_input = torch.randn(1, 3, height, width, device=device)
    output_dir = ArtifactPaths(model_config.id, args.build_root).inspector_dir
    inspect_model(
        model=model,
        example_inputs=(example_input,),
        target=board_config.dpu_fingerprint,
        device=device,
        output_dir=output_dir,
    )
    print(f"Inspector report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
