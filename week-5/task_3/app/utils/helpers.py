import yaml
import base64

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def read_config(config_path ="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def generator_simulator(content: str):
    for token in content.split(" "):
        yield token + " "