"""Parse Studio configs with an installed toolkit, without loading models or training."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from studio.domain import MODELS, normalize_project
from studio.trainer import config_for
sys.path.insert(0,sys.argv[1])
from toolkit.config import get_config
from toolkit.config_modules import ModelConfig, TrainConfig, DatasetConfig, NetworkConfig, SampleConfig, SaveConfig

for model in MODELS:
    project=normalize_project({'model':model['id'],'name':'Config check','trigger':'test_object','description':'Neutral test object'})
    cfg=config_for(project,'dataset','output','parse_test')
    config_path=root/'test-results'/('config-'+model['id']+'.json')
    config_path.write_text(json.dumps(cfg),encoding='utf-8')
    config=get_config(str(config_path))['config']['process'][0]
    parsed_model=ModelConfig(**config['model'])
    TrainConfig(**config['train'])
    DatasetConfig(**config['datasets'][0])
    NetworkConfig(**config['network'])
    SampleConfig(**config['sample'])
    SaveConfig(**config['save'])
    assert parsed_model.arch==model['arch'], parsed_model.arch
    print(f"Parsed {model['label']}: {parsed_model.arch}",flush=True)
