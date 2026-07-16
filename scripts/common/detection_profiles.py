"""
Detection profile abstraction layer.
Unifies model-specific loss retrieval, pruning excludes, and forward pass output formatting
for YOLOv5n and YOLOv26 models under a common interface.
"""
import fnmatch
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


class YOLOv26Profile(DetectionProfile):
    """
    Profile implementation for YOLOv26 (Ultralytics anchor-free end-to-end models like n, s, etc.).
    Handles E2EDetectLoss and one2one/one2many training branches.
    """
    def loss_fn(self, model):
        """
        Constructs and returns E2ELoss for YOLOv26.
        """
        # Under Vitis AI, the model class might be wrapped, so we find the inner raw Model
        # YOLOv26 in the local repository contains an end2end Detect head.
        try:
            # Import losses dynamically from local repo
            from ultralytics.utils.loss import E2ELoss, v8DetectionLoss, v8SegmentationLoss
            
            # The detect head lives at model.model[-1] in Ultralytics architecture
            detect_head = model.model[-1]

            # Mock model.args if missing or dict (since we load from raw .pt and bypass Trainer)
            from types import SimpleNamespace
            if not hasattr(model, 'args') or isinstance(model.args, dict):
                base_args = model.args if hasattr(model, 'args') and isinstance(model.args, dict) else {}
                base_args.update({'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'reg_max': 16, 'overlap_mask': False})
                model.args = SimpleNamespace(**base_args)
            elif not hasattr(model.args, 'overlap_mask'):
                model.args.overlap_mask = False

            # Initialize correct loss type (always E2ELoss for YOLOv26)
            is_seg = self.m_cfg.get('type') == 'segmentation'
            
            print(f"[INFO] Initializing YOLOv26 {'Segmentation' if is_seg else 'Detection'} loss (nc={detect_head.nc})")
            
            if is_seg:
                return E2ELoss(model, loss_fn=v8SegmentationLoss)
            else:
                return E2ELoss(model, loss_fn=v8DetectionLoss)
        except ImportError as e:
            raise ImportError(
                f"Failed to import E2ELoss or base losses from local Ultralytics repository: {e}. "
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
        Returns pruning excludes that protect the Detect/Segment26 head output
        convs (box/cls/mask/proto), whose channel counts are contractually
        fixed for DPU decoding.

        Vitis AI's ``excluded_node_names()`` only matches EXACT graph node-name
        strings or ``nn.Module`` objects - it does NOT expand globs. The config
        expresses the head output convs as dotted glob patterns (e.g.
        ``"model.22.cv2.*.2"``), which never match a node name and are silently
        ignored. Here we resolve those patterns against ``model.named_modules()``
        into the actual ``nn.Module`` objects so they are truly excluded. Exact
        XIR node names (those containing ``"::"``) are passed through unchanged.
        When ``model`` is ``None`` (e.g. early sensitivity setup) the raw config
        list is returned as-is.
        """
        raw = self.m_cfg.get('prune_excludes', [])
        if model is None:
            return list(raw)

        module_by_name = dict(model.named_modules())
        resolved = []
        for pattern in raw:
            # Exact Vitis AI graph node name: pass through untouched.
            if '::' in pattern:
                resolved.append(pattern)
                continue
            # Dotted glob over module names -> resolve to nn.Module objects.
            matches = [module_by_name[name] for name in module_by_name
                       if fnmatch.fnmatch(name, pattern)]
            if matches:
                resolved.extend(matches)
            else:
                print(f"[WARN] prune exclude pattern matched no module: {pattern}")
                resolved.append(pattern)
        return resolved

    def prepare_for_finetune(self, model):
        """
        Prepares YOLOv26 for fine-tuning by ensuring the loss-attaching components are active.
        """
        # Ensure end2end flag is enabled on the detect head so both branches are calculated
        detect_head = model.model[-1]
        detect_head.end2end = True
        detect_head.training = True
        print("[INFO] Configured Detect head module training parameters for fine-tuning.")


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
    if 'yolov26' in name or m_cfg.get('type') == 'segmentation':
        return YOLOv26Profile(m_cfg)
    elif 'yolov5n' in name:
        return YOLOv5nProfile(m_cfg)
    else:
        raise ValueError(f"No detection profile registered for model: {m_cfg['name']}")

