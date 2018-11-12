import pytest
import logging

from dwapath import DWA, DWAConfig, Rover

import os
import math
import random
import numpy as np
import pandas as pd

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

@pytest.mark.incremental
class TestObstacles(object):
    
    def test_obstacles_setup(self):
        """
        Check if the boundaries are set to obstacles
        """
        control_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.json')
        config = DWAConfig(control_file)
        dwa = DWA(config)

        expected = np.zeros((dwa.config.grid_height_len, dwa.config.grid_width_len), dtype=bool)
        expected[0, :] = expected[:, 0] = expected[dwa.config.grid_height_len - 1, :] = expected[:, dwa.config.grid_width_len - 1] = True

        pytest.config = config
        pytest.dwa = dwa
        pytest.expected = expected

        np.testing.assert_array_equal(pytest.dwa.obstacles, pytest.expected)

    def test_obstacles_set(self):
        """
        Check the set_obstacles function
        """
        df = pd.DataFrame({
            'x': [100, 200, 300, 400] , 
            'y': [400, 500, 250, 100]
            })
        pytest.dwa.set_obstacles(df['x'], df['y'])
        pytest.obstacle_df = df

        xind = (df['x'] / pytest.config.grid_reso).astype(int)
        yind = (df['y'] / pytest.config.grid_reso).astype(int)
        for i in range(len(xind)):
            pytest.expected[xind[i], yind[i]] = True

        np.testing.assert_array_equal(pytest.dwa.obstacles, pytest.expected)

    def test_obstacles_get(self):
        """
        Check the get_obstacles function
        """
        
        obs_list = (pytest.dwa.get_obstacles() / pytest.dwa.config.grid_reso).astype(int)
        x_list = np.rot90(obs_list)[1]
        y_list = np.rot90(obs_list)[0]
        obstacle_from_get_method = np.zeros((pytest.dwa.config.grid_height_len, pytest.dwa.config.grid_width_len), dtype=bool)
        obstacle_from_get_method[x_list, y_list] = True

        np.testing.assert_array_equal(obstacle_from_get_method, pytest.expected)

    def test_is_out_of_boundary(self):
        """
        Check the is_out_of_boundary function
        """
        valid_cases = pd.DataFrame({
            'x': np.random.randint(1, pytest.config.grid_height_len - 1, size=10) , 
            'y': np.random.randint(1, pytest.config.grid_width_len - 1, size=10)
            })
        
        invalid_cases = pd.DataFrame({
            'x': np.random.randint(pytest.config.grid_height_len, pytest.config.grid_height_len + 10, size=10) * (1 if random.random() < 0.5 else -1), 
            'y': np.random.randint(pytest.config.grid_width_len, pytest.config.grid_width_len + 10, size=10) * (1 if random.random() < 0.5 else -1)
            })

        valid_test = pytest.dwa.is_out_of_boundary(valid_cases['x'], valid_cases['y'])
        assert np.sum(valid_test) == 0
        
        invalid_test = pytest.dwa.is_out_of_boundary(invalid_cases['x'], invalid_cases['y'])
        assert np.sum(invalid_test) == 10

    def test_is_hit_obstacle(self):
        """
        Check the is_hit_obstacle function
        """
        
        assert np.sum(pytest.dwa.is_hit_obstacle(pytest.obstacle_df['x'], pytest.obstacle_df['y'])) == len(pytest.obstacle_df['x'])

        false_cases = pytest.dwa.is_hit_obstacle(np.array([10, 20, 30, 40]), np.array([60, 70, 80, 90]))
        np.testing.assert_array_equal(false_cases, np.array([False for _ in range(4)]))



