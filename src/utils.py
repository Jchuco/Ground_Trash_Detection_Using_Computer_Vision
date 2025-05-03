import yaml


def get_config(config_path):
    """
    Load the configuration file from the given path.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
