import torch
import torchvision.models as models
import os
import importlib.util

def load_model_skeleton(m_cfg):
    """Instantiates the raw architecture based on the source (Torchvision or Custom)."""
    source = m_cfg.get('source', 'torchvision')
    model_class = m_cfg['model_class']
    
    # --- 1. TORCHVISION SOURCE ---
    if source == 'torchvision':
        try:
            model_fn = getattr(models, model_class)
            return model_fn()
        except AttributeError:
            raise AttributeError(f"Model '{model_class}' not found in torchvision.")
    
    # --- 2. CUSTOM SOURCE (.py files) ---
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

def prepare_model(m_cfg, d_cfg, device):
    """
    Consolidated loader: Loads skeleton, adapts last layer, and loads weights.
    """
    # 1. Load Architecture
    model = load_model_skeleton(m_cfg)
    
    # 2. Dynamic Layer Replacement
    num_classes = len(d_cfg['classes'])
    last_layer_name = m_cfg.get('last_layer_name', 'fc')
    
    try:
        last_layer = getattr(model, last_layer_name)
        if isinstance(last_layer, torch.nn.Sequential):
            in_features = last_layer[-1].in_features
        else:
            in_features = last_layer.in_features
            
        setattr(model, last_layer_name, torch.nn.Linear(in_features, num_classes))
    except AttributeError:
        raise AttributeError(f"Model does not have a layer named '{last_layer_name}'. "
                             f"Check model_config.py.")

    # 3. Load Weights
    if not os.path.exists(m_cfg['model_path']):
        raise FileNotFoundError(f"Weight file not found: {m_cfg['model_path']}")
        
    checkpoint = torch.load(m_cfg['model_path'], map_location=device)
    
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # Most common: state_dict
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model