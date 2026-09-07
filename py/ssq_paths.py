"""Canonical project paths shared by all command workflows."""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
PARAMS_JSON_PATH = os.path.join(PROJECT_ROOT, 'best_params.json')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
