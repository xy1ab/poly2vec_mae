from pathlib import Path
import json
import yaml


def load_yaml_config(path):
    cfg_path = Path(path)
    with cfg_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json_config(path):
    cfg_path = Path(path)
    with cfg_path.open('r', encoding='utf-8') as f:
        return json.load(f)
