import json
from types import SimpleNamespace

import yaml
import yaml_include

# add the tag
yaml.add_constructor("!include", yaml_include.Constructor(base_dir='./configs'))

def get_config(config_file: str) -> SimpleNamespace:
    with open(config_file) as f:
        config = yaml.full_load(f)

    config = json.loads(json.dumps(config), object_hook=lambda item: SimpleNamespace(**item))

    return config
