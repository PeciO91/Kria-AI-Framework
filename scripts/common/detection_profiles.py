"""
Detection profile abstraction layer.
Unifies model-specific loss retrieval, pruning excludes, and forward pass output formatting
for YOLOv5n and YOLOv26s models under a common interface.
"""
from abc import ABC, abstractmethod
import torch

class DetectionProfile(ABC):
    """
    Abstract Base Class representing a model-specific detection workflow.
    """
    def __init__(self, m_cfg):
        self.m_cfg = m_cfg

    @abstractmethod
    def loss_fn(self, model):
        """
        Returns a callable loss function that takes (outputs, targets) and returns a scalar tensor.
        """
        pass

    @abstractmethod
    def forward_for_loss(self, model, images):
        """
        Runs a forward pass on the model configured to output what the loss function expects.
        """
        pass

    @abstractmethod
    def prune_excludes(self, model):
        """
        Returns a list of module names (glob patterns or exact matches) to exclude from channel pruning.
        """
        pass

    @abstractmethod
    def prepare_for_finetune(self, model):
        """
        Applies any modifications required to configure the model for fine-tuning.
        """
        pass


class YOLOv26sProfile(DetectionProfile):
    """
    Profile implementation for YOLOv26s (Ultralytics anchor-free end-to-end model).
    Handles E2EDetectLoss and one2one/one2many training branches.
    """
    def loss_fn(self, model):
        """
        Constructs and returns E2EDetectLoss for YOLOv26s.
        """
        # Under Vitis AI, the model class might be wrapped, so we find the inner raw Model
        # YOLOv26s in the local repository contains an end2end Detect head.
        try:
            # Import loss dynamically from local repo
            from ultralytics.utils.loss import E2EDetectLoss
            
            # The detect head lives at model.model[-1] in Ultralytics architecture
            if hasattr(model, 'model') and hasattr(model.model[-1], 'nc'):
                detect_head = model.model[-1]
            elif hasattr(model, 'module') and hasattr(model.module, 'model'):
                # Handle DP/DDP wrapping if any
                detect_head = model.module.model[-1]
            else:
                # Fallback walk through modules
                detect_head = None
                for module in model.modules():
                    if hasattr(module, 'nc') and hasattr(module, 'one2one'):
                        detect_head = module
                        break
                if detect_head is None:
                    raise RuntimeError("Could not find YOLOv26s Detect head in model structure.")

            print(f"[INFO] Initializing E2EDetectLoss with Detect head (nc={detect_head.nc}, end2end={detect_head.end2end})")
            return E2EDetectLoss(detect_head)
        except ImportError as e:
            raise ImportError(
                f"Failed to import E2EDetectLoss from local Ultralytics repository: {e}. "
                f"Ensure 'repo_path' in model_config.py points to the local ultralytics-main."
            )

    def forward_for_loss(self, model, images):
        """
        Runs the forward pass returning training-ready predictions.
        During training, YOLOv26s Detect head returns a dict with 'one2many' and 'one2one' predictions.
        """
        # Ensure model behaves in training forward mode
        was_training = model.training
        model.train()
        try:
            outputs = model(images)
        finally:
            if not was_training:
                model.eval()
        return outputs

    def prune_excludes(self, model):
        """
        Returns module name patterns that must not be pruned to maintain detection head structure.
        """
        # Return the 12 output convs configured in model_config.py
        return self.m_cfg.get('prune_excludes', [])

    def prepare_for_finetune(self, model):
        """
        Prepares YOLOv26s for fine-tuning by ensuring the loss-attaching components are active.
        """
        # Ensure end2end flag is enabled on the detect head so both branches are calculated
        for m in model.modules():
            if hasattr(m, 'nc') and hasattr(m, 'one2one'):
                m.end2end = True
                m.training = True
                print("[INFO] Configured Detect head module training parameters for fine-tuning.")
                break


class YOLOv5nProfile(DetectionProfile):
    """
    Stub profile for YOLOv5n. Since the model isn't currently retrained for DPU,
    this profile raises a clear error.
    """
    def loss_fn(self, model):
        raise NotImplementedError("YOLOv5n detection optimizer pipeline is pending model retraining.")

    def forward_for_loss(self, model, images):
        raise NotImplementedError("YOLOv5n detection optimizer pipeline is pending model retraining.")

    def prune_excludes(self, model):
        return self.m_cfg.get('prune_excludes', [])

    def prepare_for_finetune(self, model):
        raise NotImplementedError("YOLOv5n detection optimizer pipeline is pending model retraining.")


def get_profile(m_cfg):
    """
    Factory function to retrieve the correct DetectionProfile for the configured model.
    """
    name = m_cfg['name'].lower()
    if 'yolov26s' in name:
        return YOLOv26sProfile(m_cfg)
    elif 'yolov5n' in name:
        return YOLOv5nProfile(m_cfg)
    else:
        raise ValueError(f"No detection profile registered for model: {m_cfg['name']}")
