import os
import sys
import argparse

import torch

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_ROOT, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model


def _describe_output(output):
    if torch.is_tensor(output):
        return tuple(output.shape)
    if isinstance(output, (list, tuple)):
        return [_describe_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _describe_output(value) for key, value in output.items()}
    return type(output).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov26s')
    parser.add_argument('--dataset', default='coco_detection')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)
    device = torch.device(args.device)
    model = prepare_model(m_cfg, d_cfg, device)

    input_h, input_w = m_cfg['input_shape']
    dummy = torch.zeros(1, 3, input_h, input_w, device=device)
    with torch.no_grad():
        output = model(dummy)

    print('model:', m_cfg['name'])
    print('model_path:', m_cfg.get('model_path'))
    print('yaml_path:', m_cfg.get('yaml_path'))
    print('decoder:', m_cfg.get('decoder'))
    print('reg_max:', m_cfg.get('reg_max'))
    print('strides:', m_cfg.get('strides'))
    print('output_shapes:', _describe_output(output))
    if hasattr(model, 'names'):
        print('names:', model.names)
    if hasattr(model, 'stride'):
        print('model_stride:', model.stride)


if __name__ == '__main__':
    main()
