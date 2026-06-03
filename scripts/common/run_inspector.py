"""
Vitis AI inspection stage.

Runs the Vitis AI Inspector against the prepared model to verify operator
mappability to the target DPU. Produces a textual report under
build/<model>/inspector_report/ that lists each subgraph and its assigned
device. Review the report to identify any layers that would fall back to CPU.
"""
import os
import sys
import argparse

import torch
from pytorch_nndct.apis import Inspector

# Project-root import path (PROJECT_ROOT + scripts/common/ added to sys.path).
from _bootstrap import PROJECT_ROOT  # noqa: F401

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import DPU_FINGERPRINT
from model_utils import prepare_model, apply_detect_export_patch


def run_model_inspector(model_id, dataset_id):
    """Build the model and run the Vitis AI Inspector against the target DPU.

    Note: We always inspect the original (non-pruned) architecture. The
    Inspector verifies operator-level DPU compatibility, which depends on
    the topology of the network, not on channel counts. Inspecting the
    Vitis AI pruning wrapper would also fail because its internal graph
    forbids deepcopy (which the Inspector uses internally).
    """
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)

    output_dir = os.path.join("build", m_cfg['name'].lower(), "inspector_report")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Starting Inspector for: {m_cfg['name']} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Always inspect the original architecture (prune_threshold=None).
    model = prepare_model(m_cfg, d_cfg, device, prune_threshold=None)

    # Inspect the same graph that gets quantized and compiled: for Ultralytics
    # end-to-end detection heads this exports the one2one branches and drops the
    # in-graph DFL decode + topk post-processing that never reach the DPU.
    # No-op for classification / segmentation models.
    num_patched = apply_detect_export_patch(model)
    if num_patched:
        print(f"[INFO] Applied detection export patch to {num_patched} "
              f"head(s); inspecting the deployed one2one graph.")

    inspector = Inspector(DPU_FINGERPRINT)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn(1, 3, input_h, input_w).to(device)

    print(f"[INFO] Inspecting with input shape: {dummy_input.shape}...")
    inspector.inspect(model, (dummy_input,), device=device, output_dir=output_dir)
    print(f"=== Inspection Complete. Report saved to {output_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model ID. Falls back to ACTIVE_MODEL_ID '
                             'in model_config.py when omitted.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset ID. Falls back to ACTIVE_DATASET_ID '
                             'in dataset_config.py when omitted.')
    args = parser.parse_args()

    run_model_inspector(args.model, args.dataset)
