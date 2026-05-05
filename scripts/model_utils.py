"""
Host-side model preparation helpers shared by the inspector, optimizer and
quantizer scripts.

`load_model_skeleton` instantiates a model architecture from either
torchvision or a custom Python file (with a special path-injection branch
for YOLOv5 so its internal `models/` and `utils/` packages resolve correctly).

`prepare_model` chains skeleton instantiation, optional last-layer adaptation
(for classification tasks), optional Vitis AI structural pruning, and weight
loading. The returned model is on `device` and in eval mode.
"""
import os
import sys
import importlib.util

import torch
import torch.nn as nn
import torchvision.models as models

# Pruner is only available inside the Vitis AI Docker image.
try:
    from pytorch_nndct.apis import Pruner
    HAS_PRUNER = True
except ImportError:
    HAS_PRUNER = False


def load_model_skeleton(m_cfg):
    """
    Instantiate the raw model architecture for `m_cfg`.

    Supports two sources:
      - 'torchvision': any model exposed under `torchvision.models`.
      - 'custom':       a Python file whose top-level defines `m_cfg['model_class']`.

    YOLOv5 is treated specially because its module imports rely on the
    package being importable from `models/yolov5n/` and the constructor
    expects a YAML configuration path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    source = m_cfg.get('source', 'torchvision')
    model_class = m_cfg['model_class']

    if source == 'torchvision':
        try:
            model_fn = getattr(models, model_class)
            return model_fn()
        except AttributeError:
            raise AttributeError(f"Model '{model_class}' not found in torchvision.")

    if source != 'custom':
        raise ValueError(f"Unknown source: {source}. Use 'torchvision' or 'custom'.")

    file_path = m_cfg.get('file_path')
    abs_file_path = os.path.join(project_root, file_path) if file_path and not os.path.isabs(file_path) else file_path
    if not abs_file_path or not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"Custom model file not found at: {abs_file_path}")

    if m_cfg.get('loader') == 'ultralytics':
        repo_path = m_cfg.get('repo_path')
        abs_repo_path = os.path.join(project_root, repo_path) if repo_path and not os.path.isabs(repo_path) else repo_path
        weights_path = m_cfg.get('model_path')
        abs_weights_path = os.path.join(project_root, weights_path) if weights_path and not os.path.isabs(weights_path) else weights_path
        if not abs_repo_path or not os.path.exists(abs_repo_path):
            raise FileNotFoundError(f"Ultralytics repo not found at: {abs_repo_path}")
        if not abs_weights_path or not os.path.exists(abs_weights_path):
            raise FileNotFoundError(f"Weight file not found: {abs_weights_path}")

        module_name = os.path.basename(abs_file_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, model_class)(
            weights_path=abs_weights_path,
            repo_root=abs_repo_path,
            head_variant=m_cfg.get('head_variant', 'one2one'),
            replace_leaky_relu=m_cfg.get('replace_leaky_relu', False),
        )

    # YOLOv5 needs its package root on sys.path and a YAML config to instantiate.
    if m_cfg.get('loader') == 'yolov5' or ("yolo" in m_cfg['name'].lower() and m_cfg.get('loader') is None):
        yolo_root = os.path.join(project_root, 'models', 'yolov5n')
        if yolo_root not in sys.path:
            sys.path.append(yolo_root)

        cfg_path = m_cfg.get('yaml_path',
                             os.path.join(yolo_root, 'models', 'yolov5n.yaml'))
        cfg_path = os.path.join(project_root, cfg_path) if not os.path.isabs(cfg_path) else cfg_path
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"YAML config not found at: {cfg_path}")

        spec = importlib.util.spec_from_file_location("yolo", abs_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, model_class)(cfg=cfg_path)

    # Generic custom loader (e.g. UNet).
    module_name = os.path.basename(abs_file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_class)()


def prepare_model(m_cfg, d_cfg, device, prune_threshold=None):
    """
    Build a deployment-ready model in four stages.

    1. Instantiate the architecture skeleton.
    2. For classification, replace the final layer with one sized to the
       active dataset's class count.
    3. If a pruned weight file exists *and* `prune_threshold` is provided,
       slim the architecture with the Vitis AI Pruner.
    4. Load weights (with smart extraction for YOLOv5 checkpoints) and
       move the model to `device` in eval mode.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # 1. Architecture
    model = load_model_skeleton(m_cfg)

    if m_cfg.get('weights_loaded_by_wrapper'):
        model.to(device)
        model.eval()
        return model

    # 2. Last-layer adaptation for classification.
    if m_cfg.get('type') == 'classification':
        num_classes = len(d_cfg['classes'])
        last_layer_name = m_cfg.get('last_layer_name', 'fc')
        try:
            last_layer = getattr(model, last_layer_name)
        except AttributeError:
            raise AttributeError(f"Model does not have a layer named '{last_layer_name}'.")

        if isinstance(last_layer, nn.Sequential):
            in_features = last_layer[-1].in_features
            last_layer[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = last_layer.in_features
            setattr(model, last_layer_name, nn.Linear(in_features, num_classes))

    # 2b. Detection: the YOLOv5 Detect head is already stripped in models/yolov5n/yolo.py
    # so the network returns the three raw P3/P4/P5 conv outputs.

    # 3. Optional structural pruning.
    target_weight_path = m_cfg['model_path']
    pruned_weight_path = target_weight_path.replace(".pt", "_pruned.pt")
    abs_pruned_path = os.path.join(project_root, pruned_weight_path)

    if os.path.exists(abs_pruned_path) and prune_threshold is not None and HAS_PRUNER:
        print(f"[INFO] Pruned weights detected. Slimming architecture (ratio: {prune_threshold})")
        input_h, input_w = m_cfg['input_shape']
        dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)
        pruner = Pruner(model, dummy_input)
        model = pruner.prune(threshold=prune_threshold)
        target_weight_path = pruned_weight_path

    # 4. Load weights.
    abs_weight_path = os.path.join(project_root, target_weight_path)
    if not os.path.exists(abs_weight_path):
        raise FileNotFoundError(f"Weight file not found: {abs_weight_path}")

    print(f"[INFO] Loading weights from: {abs_weight_path}")
    checkpoint = torch.load(abs_weight_path, map_location=device)

    # YOLOv5 checkpoints wrap the state_dict inside a 'model' key.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = (checkpoint['model'].state_dict()
                      if hasattr(checkpoint['model'], 'state_dict')
                      else checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    # strict=False is required because the YOLO Detect head was stripped.
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
