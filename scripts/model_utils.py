import torch
import torch.nn as nn
import torchvision.models as models
import os
import importlib.util

# Pruner is only available on the Host/Docker side (NNDCT)
try:
    from pytorch_nndct.apis import Pruner
    HAS_PRUNER = True
except ImportError:
    HAS_PRUNER = False

def load_model_skeleton(m_cfg):
    """Instantiates the raw architecture based on the source (Torchvision or Custom)."""
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
            
        module_name = os.path.basename(file_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        try:
            return getattr(module, model_class)()
        except AttributeError:
            raise AttributeError(f"Class '{model_class}' not found in file '{file_path}'.")
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'torchvision' or 'custom'.")

def prepare_model(m_cfg, d_cfg, device, prune_threshold=None):
    """
    Consolidated loader: 
    1. Loads skeleton 
    2. Adapts last layer (ONLY for classification)
    3. Handles Pruning (Slimming architecture via provided threshold)
    4. Loads weights via absolute paths
    """
    # 0. Setup Absolute Paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # 1. Load Architecture
    if m_cfg.get('name', '').lower().startswith('yolo'):
        # Add the yolov5 git folder to python path so imports inside it work
        yolo_path = os.path.join(project_root, 'yolov5') # Assumes folder is named 'yolov5'
        if yolo_path not in sys.path:
            sys.path.append(yolo_path)
        
        from models.common import DetectMultiBackend
        print(f"[INFO] Loading YOLOv5 model via DetectMultiBackend...")
        # This loads the YAML and Weights automatically
        model = DetectMultiBackend(weights=os.path.join(project_root, m_cfg['model_path']), device=device)
        model = model.model # Get the underlying nn.Module
    else:
        # Standard loading for ResNet/MobileNet
        model = load_model_skeleton(m_cfg)
    
    # 2. Dynamic Layer Replacement
    # Only do this if the model is a classification model!
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
    else:
        print(f"[INFO] {m_cfg.get('type').capitalize()} model detected. Skipping Linear layer replacement.")

    # 3. Handle Pruning (Optimizer logic)
    target_weight_path = m_cfg['model_path']
    pruned_weight_path = target_weight_path.replace(".pt", "_pruned.pt")
    abs_pruned_path = os.path.join(project_root, pruned_weight_path)

    if os.path.exists(abs_pruned_path) and prune_threshold is not None:
        if not HAS_PRUNER:
            print("[WARN] Pruned weights detected but Pruner (NNDCT) not available.")
        else:
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
        
    checkpoint = torch.load(abs_weight_path, map_location=device)
    
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # strict=False is safer for YOLO models which often have slight key mismatches in the Detect head
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    return model