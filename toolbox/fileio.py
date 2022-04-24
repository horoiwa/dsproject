from pathlib import Path
import json


def save_dict_as_json(d, out_path):
    for key in d.keys():
        d[key] = str(d[key])
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
