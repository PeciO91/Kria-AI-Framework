"""
Shared sys.path bootstrap for all stage scripts.

Every host-side script (deploy / inspector / quantizer / optimizer / compiler)
and every task-specific runner imports this module at the top so that:

  * The project root (where ``model_config.py``, ``dataset_config.py`` and
    ``board_config.py`` live) is on ``sys.path``.
  * ``scripts/common/`` is on ``sys.path`` so peer modules
    (``model_utils``, ``board_utils``, ``dataset_utils``, ``optimizer_utils``,
    ``detection_profiles``) can be imported by bare name regardless of which
    sub-folder the caller lives in.

Layout assumed::

    Project/
        model_config.py
        dataset_config.py
        board_config.py
        scripts/
            common/   _bootstrap.py  (this file)
            classification/
            detection/
            segmentation/
"""
import os
import sys

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(COMMON_DIR, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_ROOT, os.pardir))

for _p in (PROJECT_ROOT, COMMON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
