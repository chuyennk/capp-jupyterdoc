import json
import os

def load_configs(modelname):
    config_file = os.environ.get('HOME_DIR') + f"/models/{modelname}/data/configs.json"
    return json.load(open(config_file))

def get_config(configs, configname):
    return configs.get(configname)    