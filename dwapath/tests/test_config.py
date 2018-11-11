
from dwapath import DWAConfig, DWA

import json
import os
import math

def test_config():
    """
    Check if DWAConfig loads the control file correctly
    """
    control_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.json')

    config = DWAConfig(control_file)
    with open(control_file) as f:
        expected = json.load(f)
        expected['max_yawrate'] *= math.pi / 180.0
        expected['max_dyawrate'] *= math.pi / 180.0
        expected['yawrate_reso'] *= math.pi / 180.0
        expected['grid_width_len'] = int(expected['grid_width'] // expected['grid_reso'] + 1)
        expected['grid_height_len'] = int(expected['grid_height'] // expected['grid_reso'] + 1)
    
    assert expected == config.__dict__
    