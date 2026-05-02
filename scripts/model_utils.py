import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys
import importlib.util

# Pruner is only available on the Host/Docker side (NNDCT)
try:
    from pytorch_nndct.apis import Pruner
    HAS_PRUNER = True
except ImportError:
    HAS_PRUNER = False

def load_model_skeleton(m_cfg):
    """Instantiates the raw architecture based on the source (Torchvision or Custom)."""
    # 0. Setup Absolute Paths
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
    
    elif source == 'custom':
        file_path = m_cfg.get('file_path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom model file not found at: {file_path}")
        
        # --- YOLO SPECIFIC PATH INJECTION ---
        if "yolo" in m_cfg['name'].lower():
            # Add yolov5n directory to path for imports (includes models/, utils/, etc.)
            yolo_root = os.path.join(project_root, 'models', 'yolov5n')
            if yolo_root not in sys.path:
                sys.path.append(yolo_root)
            
            # YAML path needed for DetectionModel initialization
            cfg_path = m_cfg.get('yaml_path', os.path.join(yolo_root, 'models', 'yolov5n.yaml'))
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"YAML config not found at: {cfg_path}")
            
            # Load the module specifically
            spec = importlib.util.spec_from_file_location("yolo", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Return the model class (usually DetectionModel) with the cfg file
            try:
                return getattr(module, model_class)(cfg=cfg_path)
            except Exception as e:
                print(f"[ERROR] Failed to instantiate {model_class}: {e}")
                raise e

        # Standard custom loader for other models (e.g. UNet)
        module_name = os.path.basename(file_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return getattr(module, model_class)()
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'torchvision' or 'custom'.")

def prepare_model(m_cfg, d_cfg, device, prune_threshold=None):
    """
    Consolidated loader: 
    1. Loads skeleton 
    2. Adapts last layer (ONLY for classification)
    3. Handles Pruning
    4. Loads weights via absolute paths
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # 1. Load Architecture
    model = load_model_skeleton(m_cfg)
    
    # 2. Dynamic Layer Replacement
    if m_cfg.get('type') == 'classification':
        num_classes = len(d_cfg['classes'])
        last_layer_name = m_cfg.get('last_layer_name', 'fc')
        
        try:
            last_layer = getattr(model, last_layer_name)
            if isinstance(last_layer, torch.nn.Sequential):
                in_features = last_layer[-1].in_features
                last_layer[-1] = torch.nn.Linear(in_features, num_classes)
            else:
                in_features = last_layer.in_features
                setattr(model, last_layer_name, torch.nn.Linear(in_features, num_classes))
        except AttributeError:
            raise AttributeError(f"Model does not have a layer named '{last_layer_name}'.")
    
    # 2b. YOLO Detect Head - User has already stripped it in yolo.py
    # The Detect.forward now only runs conv layers and returns raw outputs
    # No additional stripping needed

    # 3. Handle Pruning
    target_weight_path = m_cfg['model_path']
    pruned_weight_path = target_weight_path.replace(".pt", "_pruned.pt")
    abs_pruned_path = os.path.join(project_root, pruned_weight_path)

    if os.path.exists(abs_pruned_path) and prune_threshold is not None:
        if HAS_PRUNER:
            print(f"[INFO] Pruned weights detected. Slimming architecture (Ratio: {prune_threshold})")
            input_h, input_w = m_cfg['input_shape']
            dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)
            pruner = Pruner(model, dummy_input)
            model = pruner.prune(threshold=prune_threshold) 
            target_weight_path = pruned_weight_path

    # 4. Load Final Weights
    abs_weight_path = os.path.join(project_root, target_weight_path)
    if not os.path.exists(abs_weight_path):
        raise FileNotFoundError(f"Weight file not found: {abs_weight_path}")
        
    print(f"[INFO] Loading weights from: {abs_weight_path}")
    checkpoint = torch.load(abs_weight_path, map_location=device)
    
    # Smart weight extraction for YOLOv5 dictionary format
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        
    # strict=False is required because the Detect head has been stripped in the code
    model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model