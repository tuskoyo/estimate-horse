import configparser
import os


def read_config():
    config = configparser.ConfigParser()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(file_dir, '..', '..', 'config.ini'))
    return config

def get_config_path():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(file_dir, '..', '..', 'config.ini')

def get_config_value(section, key):
    config = read_config()
    return os.path.expanduser(config[section][key])

if __name__ == '__main__':
    print(get_config_value('model-setting', 'pose_model'))
