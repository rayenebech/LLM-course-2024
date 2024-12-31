from pathlib import Path
import yaml

def load_files(root_dir):
    files = []
    root_path = Path(root_dir).resolve()
    files_generator = root_path.rglob('*.pdf')
    for file in files_generator:
        files.append(str(file))
    return files
    
def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
