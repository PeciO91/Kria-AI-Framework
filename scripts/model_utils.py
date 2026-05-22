"""
Host-side model preparation helpers shared by the inspector, optimizer and
quantizer scripts.

`load_model_skeleton` instantiates a model architecture from torchvision,
a custom Python file, a local Ultralytics repository, or a local YOLO
repository.

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

from pytorch_nndct import IterativePruningRunner

# Compute project paths once at module load time
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


def normalize_path(path):
    """Convert relative path to absolute path using project_root.
    If path is already absolute, return it as-is.
    """
    if path and not os.path.isabs(path):
        return os.path.join(project_root, path)
    return path


def add_repo_to_syspath(repo_path):
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)


def clear_module_tree(root_names):
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(f"{root}.")
               for root in root_names):
            del sys.modules[module_name]


def derive_weight_path(model_path, suffix):
    """Append `suffix` to `model_path` while preserving the original
    file extension. Examples:
      derive_weight_path("models/foo.pt",  "_pruned") -> "models/foo_pruned.pt"
      derive_weight_path("models/foo.pth", "_slim")   -> "models/foo_slim.pth"
    """
    base, ext = os.path.splitext(model_path)
    return f"{base}{suffix}{ext}"


def extract_state_dict(checkpoint):
    """Extract state_dict from various checkpoint formats.
    Handles Ultralytics-style, standard PyTorch, and raw state_dict checkpoints.
    """
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Ultralytics-style: checkpoint['model'] contains the model or state_dict
        sd = (checkpoint['model'].state_dict()
              if hasattr(checkpoint['model'], 'state_dict')
              else checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Standard PyTorch: checkpoint['state_dict']
        sd = checkpoint['state_dict']
    else:
        # Raw state_dict or model object
        sd = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    return sd


def load_model_skeleton(m_cfg):
    """
    Instantiate a model architecture skeleton (no weights) based on the
    configuration in `model_config.py`.

    Supports torchvision models, generic custom Python files, local
    Ultralytics repositories, and local YOLO repositories.
    """
    source = m_cfg.get('source', 'torchvision')
    loader = m_cfg.get('loader')

    if source == 'torchvision':
        model_class = m_cfg['model_class']
        try:
            model_fn = getattr(models, model_class)
            return model_fn()
        except AttributeError:
            raise AttributeError(f"Model '{model_class}' not found in torchvision.")

    if source != 'custom':
        raise ValueError(f"Unknown source: {source}. Use 'torchvision' or 'custom'.")

    if loader == 'ultralytics':
        abs_repo_path = normalize_path(m_cfg.get('repo_path'))
        abs_weights_path = normalize_path(m_cfg.get('model_path'))
        if not abs_repo_path or not os.path.exists(abs_repo_path):
            raise FileNotFoundError(f"Ultralytics repo not found at: {abs_repo_path}")
        if not abs_weights_path or not os.path.exists(abs_weights_path):
            raise FileNotFoundError(f"Weight file not found: {abs_weights_path}")

        add_repo_to_syspath(abs_repo_path)
        clear_module_tree(('ultralytics',))
        from ultralytics import YOLO
        yolo = YOLO(abs_weights_path)
        return yolo.model

    # YOLO (v5, v6, v3, etc.) needs its package root on sys.path and a YAML config to instantiate.
    if loader == 'yolo' or ("yolo" in m_cfg['name'].lower() and loader is None):
        abs_repo_path = normalize_path(m_cfg.get('repo_path'))
        if not abs_repo_path or not os.path.exists(abs_repo_path):
            raise FileNotFoundError(f"YOLO repo not found at: {abs_repo_path}")
        add_repo_to_syspath(abs_repo_path)

        cfg_path = m_cfg.get('yaml_path',
                             os.path.join(abs_repo_path, 'models', f"{m_cfg['name'].lower()}.yaml"))
        cfg_path = normalize_path(cfg_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"YAML config not found at: {cfg_path}")

        clear_module_tree(('models', 'utils'))
        from models.yolo import Model
        return Model(cfg=cfg_path)

    # Generic custom loader (e.g. UNet).
    model_class = m_cfg['model_class']
    abs_file_path = normalize_path(m_cfg.get('file_path'))
    if not abs_file_path or not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"Custom model file not found at: {abs_file_path}")

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
       configured class count.
    3. If a pruned weight file exists *and* `prune_threshold` is provided,
       slim the architecture with the Vitis AI Pruner.
    4. Load weights (with smart extraction for common checkpoint formats) and
       move the model to `device` in eval mode.
    """
    # 1. Architecture
    model = load_model_skeleton(m_cfg)

    # 2. Classification: adapt last layer to match the class count.
    # Model config can override the dataset's class count/list (e.g. when a
    # model was trained on a subset of the dataset's classes).
    if m_cfg.get('type') == 'classification':
        last_layer_name = m_cfg.get('last_layer_name', 'fc')
        try:
            last_layer = getattr(model, last_layer_name)
        except AttributeError:
            raise AttributeError(f"Model does not have a layer named '{last_layer_name}'.")

        # num_classes priority: model_config > model_config['classes'] > dataset_config['classes']
        # Use 0 in model_config to explicitly fall back to dataset classes.
        num_classes = m_cfg.get('num_classes', 0)
        if num_classes == 0:
            num_classes = len(m_cfg.get('classes', d_cfg.get('classes', [])))
        print(f"[INFO] Configuring last layer for {num_classes} classes")
        if isinstance(last_layer, nn.Sequential):
            in_features = last_layer[-1].in_features
            last_layer[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = last_layer.in_features
            setattr(model, last_layer_name, nn.Linear(in_features, num_classes))

    # 3. Optional structural pruning.
    target_weight_path = m_cfg['model_path']
    pruned_weight_path = derive_weight_path(target_weight_path, "_pruned")
    abs_pruned_path = normalize_path(pruned_weight_path)

    pruning_applied = False
    if os.path.exists(abs_pruned_path) and prune_threshold is not None:
        # Works for both --method iterative AND --method onestep produced
        # weights: both runners use the same .vai/<Model>_ratio_<R>.spec output
        # and share IterativePruningRunner.prune() semantics for re-slimming.
        print(f"[INFO] Pruned weights detected. Slimming architecture (ratio: {prune_threshold})")
        input_h, input_w = m_cfg['input_shape']
        dummy_input = torch.randn([1, 3, input_h, input_w], dtype=torch.float32).to(device)
        runner = IterativePruningRunner(model, dummy_input)
        excludes = m_cfg.get('prune_excludes', [])
        if excludes:
            print(f"[INFO] Retaining excluded layers during reconstruction: {excludes}")
        model = runner.prune(removal_ratio=prune_threshold, excludes=excludes)
        target_weight_path = pruned_weight_path
        pruning_applied = True
    elif prune_threshold is not None and not os.path.exists(abs_pruned_path):
        print(f"[WARN] prune_threshold={prune_threshold} given but {abs_pruned_path} not found.")
        print(f"[WARN] Run the optimizer stage first to produce pruned weights.")

    # 4. Load weights.
    abs_weight_path = normalize_path(target_weight_path)
    if not os.path.exists(abs_weight_path):
        raise FileNotFoundError(f"Weight file not found: {abs_weight_path}")

    print(f"[INFO] Loading weights from: {abs_weight_path}")
    checkpoint = torch.load(abs_weight_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    # Defensive shape check: detect slim-vs-full mismatch and report clearly.
    if pruning_applied:
        model_sd = model.state_dict()
        sample_key = next((k for k in state_dict if k in model_sd
                           and state_dict[k].shape != model_sd[k].shape), None)
        if sample_key:
            raise RuntimeError(
                f"Shape mismatch after pruning: checkpoint '{sample_key}' has "
                f"shape {tuple(state_dict[sample_key].shape)} but model has "
                f"shape {tuple(model_sd[sample_key].shape)}. "
                f"This means runner.prune() did not slim the model to match the saved "
                f"_pruned.pt. Likely the .vai/{model.__class__.__name__}.sens cache is "
                f"stale or missing. Re-run the optimizer with --mode all to regenerate."
            )

    # Custom detection graphs may intentionally omit or reshape some checkpoint keys.
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
