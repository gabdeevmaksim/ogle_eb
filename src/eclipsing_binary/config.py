import configparser
import os

def get_paths():
    """Reads data paths from the configuration file."""

    config = configparser.ConfigParser()
    config.read('../config.ini')

    paths = config['paths']
    base_dir = paths['w_dir']  # Get the base directory

    return {
        'w_dir': base_dir,
        'demo_data_dir': os.path.join(base_dir, paths['demo_data_dir']),
        'ecl_file': os.path.join(base_dir, paths['demo_data_dir'], paths['ecl_file']),
        'ident_file': os.path.join(base_dir, paths['demo_data_dir'], paths['ident_file']),
        'lc_I_dir': os.path.join(base_dir, paths['demo_data_dir'], paths['lc_I_dir']),
        'lc_V_dir': os.path.join(base_dir, paths['demo_data_dir'], paths['lc_V_dir'])
    }