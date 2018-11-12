"""
Mobile robot motion planning sample with Dynamic Window Approach
author: Chia-Hua 'Kai' Peng (@KaiPeng21)
"""

import json
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from dwapath.utils.timer import TaskTimer

class DWAConfig:

    def __init__(self, control_file):

        with open(control_file) as f:
            self.__dict__ = json.load(f)
            self.max_yawrate *= math.pi / 180.0
            self.max_dyawrate *= math.pi / 180.0
            self.yawrate_reso *= math.pi / 180.0
            self.grid_width_len = int(self.grid_width / self.grid_reso + 1)
            self.grid_height_len = int(self.grid_height / self.grid_reso + 1)

class Rover:

    def __init__(self, x, y, yaw, v=0.0, omega=0.0):
        """
        :param x: the initial x coordinate of the rover (cm)
        :param y: the initial y coordinate of the rover (cm)
        :param yaw: the initial heading of the rover (rad)
        :param v: the initial velocity of the rover (cm/s)
        :param omega: the initial yawrate of the rover (rad/s)
        :rtype: A :class:`Rover <Rover>`
        """
        self.x = x
        self.y = y
        self.yaw = yaw * math.pi / 180
        self.v = v
        self.omega = omega * math.pi / 180
    
    def __repr__(self):
        return '<Rover x: %8.2f , y: %8.2f, yaw: %6.3f, v: %6.2f, w: %6.3f>' % (self.x, self.y, self.yaw * 180 / math.pi, self.v, self.omega * 180 / math.pi)

    def motion(self, u_velocity, u_yawrate, delta_time):
        """ Move the rover by setting the target velocity and target yawrate
        :param u_velocity: target velocity (cm/s)
        :param u_yawrate: target u_yawrate (rad/s)
        :param delta_time: time difference (s)
        :rtype: None
        """
        self.x += self.v * math.cos(self.yaw) * delta_time
        self.y += self.v * math.sin(self.yaw) * delta_time
        self.yaw += u_yawrate * delta_time
        self.v = u_velocity
        self.omega = u_yawrate

class DWA:

    def __init__(self, config):
        """ Robotic Path Finding Using Dynamic Windows Approach
        :param config: DWA Configure Class `<DWAConfig>` 
        :rtype: A :class:`DWA <DWA>`
        """
        self.config = config
        self.obstacles = np.zeros((config.grid_height_len, config.grid_width_len)).astype(bool)

        # set the boundary obstacles
        self.obstacles[:, 0] = self.obstacles[0, :] = self.obstacles[-1, :] = self.obstacles[:, -1] = True
    
    def __repr__(self):
        return '<DWA>'

    def get_obstacles(self):
        """ Get a 2d numpy array of obstacles in the form [[x, y], [x, y], ...]
        :rtype: A numpy array of obstacle indices
        """
        return np.argwhere(self.obstacles == True) * self.config.grid_reso

    def set_obstacles(self, x, y):
        """ Set obstacle at specific indecis
        :param x: obstacle x position (cm)
        :param y: obstacle y position (cm)
        :rtype: None
        """
        try:
            self.obstacles[(x / self.config.grid_reso).astype(int), (y / self.config.grid_reso).astype(int)] = True
        except:
            self.obstacles[int(x / self.config.grid_reso), int(y / self.config.grid_reso)] = True


    def is_out_of_boundary(self, x, y):
        """ Check if the given x, y (in cm) is out of bound
        :param x: numpy array or dataframe series of x coordinates
        :param y: numpy array or dataframe series of y coordinates
        :rtype: numpy array or dataframe series of booleans
        """
        return (x <= 0) | (x >= self.config.grid_width) | (y <= 0) | (y >= self.config.grid_height)        

    def is_hit_obstacle(self, x, y):
        """ Check if the given x, y (in cm) hit a obstacle
        :param x: numpy array or dataframe series of x coordinates
        :param y: numpy array or dataframe series of y coordinates
        :rtype: numpy array or dataframe series of booleans
        """
        x = (x / self.config.grid_reso).astype(int)
        y = (y / self.config.grid_reso).astype(int)
        # assuming [0, 0] is always an obstacle because of the boundary
        return self.obstacles[x * ~self.is_out_of_boundary(x, y), y * ~self.is_out_of_boundary(x, y)]

    def _calc_dynamic_window(self, rover):
        """ Compute the dynamic window, or the velocity and yawrate ranges in the next iteration
        :param rover: a Rover Class `<Rover>`
        :rtype: A dynamic window tuple in the form `(v_min, v_max, yawrate_min, yawrate_max)`
        """
        v_min = max(self.config.min_speed, rover.v - self.config.max_accel * self.config.dt)
        v_max = min(self.config.max_speed, rover.v + self.config.max_accel * self.config.dt)
        yawrate_min = max(-self.config.max_yawrate, rover.omega - self.config.max_dyawrate * self.config.dt)
        yawrate_max = min(self.config.max_yawrate, rover.omega + self.config.max_dyawrate * self.config.dt)
        return v_min, v_max, yawrate_min, yawrate_max

    def _create_dw_dataframe(self, rover, dynamic_window):
        """ Create a dataframe of dynamic window using ranges of velocity, yawrate, and time
        :param rover: a Rover Class `<Rover>`
        :param dynamic_window: 
        :rtype: A dataframe class `<pandas.DataFrame>`
        """
        # get the velocity, yawrate, predic timestamps from the dynamic window and config
        velocity_range = np.arange(dynamic_window[0], dynamic_window[1] + self.config.v_reso, self.config.v_reso)
        yawrate_range = np.arange(dynamic_window[2], dynamic_window[3] + self.config.yawrate_reso, self.config.yawrate_reso)
        time_range = np.arange(self.config.dt, self.config.predict_time + self.config.dt, self.config.dt)

        # get the number of possible trajectories
        num_possible_traj = len(velocity_range) * len(yawrate_range)

        # construct an iter product of the velocity, yawrate, and time ranges np arrays
        prod = np.stack(np.meshgrid(velocity_range, yawrate_range, time_range), -1).reshape(-1, 3)
        prod = np.rot90(prod)

        # construct the dataframe
        df = pd.DataFrame({'time' : prod[0], 'u_yawrate' : prod[1] , 'u_velocity' : prod[2]})

        # predict the trajectory from the current rover position and possible dynamic window
        # predicte yaw at time df['time']
        df['predict_yaw'] = rover.yaw + df['u_yawrate'] * df['time']
        
        # compute yaw at time n by shifting the predicted yaw by 1 row
        # yaw_0 = rover.yaw
        # yaw_1 = rover.yaw + yawrate * 1
        # yaw_2 = rover.yaw + yawrate * 2
        # yaw_n = rover.yaw + yawrate * n
        df['yaw_n'] = df['predict_yaw'].shift(1)
        df.loc[df['time'] == self.config.dt, 'yaw_n'] = rover.yaw

        # compute offset from current rover position after n seconds
        # x_1 = rover.x + V*cos(yaw_0)*dt
        # x_2 = rover.x + V*cos(yaw_0)*dt + V*cos(yaw_1)*dt
        # x_3 = rover.x + V*cos(yaw_0)*dt + V*cos(yaw_1)*dt + V*cos(yaw_2)*dt 
        # x_n = rover.x + Sigma{ V*cos(yaw_k)*dt } k=[0...n]

        df['V-cos(yaw_n)-delta_t'] = df['u_velocity'] * np.cos(df['yaw_n']) * self.config.dt
        tmp = np.reshape(df['V-cos(yaw_n)-delta_t'].values, (num_possible_traj, len(df['V-cos(yaw_n)-delta_t']) // num_possible_traj))
        tmp = np.cumsum(tmp, axis=1)
        df['Accum(V-cos(yaw)-delta_t)'] = tmp.flatten()
        
        df['V-sin(yaw_n)-delta_t'] = df['u_velocity'] * np.sin(df['yaw_n']) * self.config.dt
        tmp = np.reshape(df['V-sin(yaw_n)-delta_t'].values, (num_possible_traj, len(df['V-sin(yaw_n)-delta_t']) // num_possible_traj))
        tmp = np.cumsum(tmp, axis=1)
        df['Accum(V-sin(yaw)-delta_t)'] = tmp.flatten()

        # predict x and y positions
        df['predict_x'] = rover.x + df['Accum(V-cos(yaw)-delta_t)']
        df['predict_y'] = rover.y + df['Accum(V-sin(yaw)-delta_t)']
        
        df['predict_id'] = df.index.values

        # drop the unecessary columns
        df.drop(['yaw_n', 'V-cos(yaw_n)-delta_t', 'V-sin(yaw_n)-delta_t', 'Accum(V-cos(yaw)-delta_t)', 'Accum(V-sin(yaw)-delta_t)'], axis=1, inplace=True)

        return df

    def _calc_possible_input_costs(self, df, rover, goal):
        """ Compute the cost for all predicted trajectory
        :param df: the dynamic window dataframe creared from _create_dw_dataframe
        :param rover: a Rover Class `<Rover>` 
        :param goal: a tuple in the form `(x, y)`
        :rtype: A dataframe class `<pandas.Dataframe>`
        """

        gx, gy = goal

        # Compute heading cost
        # - compute the error angle from 
        # - angle = acos((RT * RG) / (|RT| * |RG|)) given R: Rover Position, T: Predict Trajectory, G: Goal
        # - the smaller the angle -> the smaller the cost
        # magnitude_goal = ((gx - rover.x)**2 + (gy - rover.y)**2)**0.5
        # print('magnitude_goal: ', magnitude_goal)
        # df['magnitude_traj'] = ((df['predict_x'] - rover.x)**2 + (df['predict_y'] - rover.y)**2)**0.5
        # df['dot_product'] = (gx - rover.x)*(df['predict_x'] - rover.x) + (gy - rover.y)*(df['predict_y'] - rover.y)
        # df['error'] = df['dot_product'] / (magnitude_goal * df['magnitude_traj'])
        # df['error_angle'] = np.arccos((df['error'] > 1) + (df['error'] <= 1)*df['error'])
        # df['cost_heading'] = self.config.cost_gain_heading * df['error_angle'] / math.pi
        # df.drop(['magnitude_traj', 'dot_product', 'error', 'error_angle'], axis=1, inplace=True)
        
        # magnitude_goal = (gx**2 + gy**2)**0.5
        # df['magnitude_traj'] = (df['predict_x']**2 + df['predict_y']**2)**0.5
        # df['dot_product'] = gx*

 
        # Compute velocity cost
        # - compute how fast the rover is moving
        # - the faster the velocity -> the smaller the cost
        df['cost_velocity'] = (self.config.max_speed - df['u_velocity']) / (self.config.max_speed - df['u_velocity'].mean() + 1) * self.config.cost_gain_velocity

        # Compute distance cost
        dx_mean = df['predict_x'].mean()
        dy_mean = df['predict_y'].mean()
        df['cost_distance'] = ((gx - df['predict_x'])**2 + (gy - df['predict_y'])**2) / ((gx - dx_mean)**2 + (gy - dy_mean)**2 + 1) * self.config.cost_gain_distance
        
        # Compute obstacle cost
        #obstacle_list = np.rot90(self.get_obstacles())
        #obstacle_list_x = obstacle_list[1]
        #obstacle_list_y = obstacle_list[0]
        #obstacle_df = pd.DataFrame({'ox': obstacle_list_x, 'oy': obstacle_list_y})
        #obstacle_df['tmp'] = 1
        #df['tmp'] = 1
        #obstacle_df = pd.merge(obstacle_df, df, on='tmp', how='outer')
        #obstacle_df['obstacle_dist_square'] = (obstacle_df['predict_x'] - obstacle_df['ox'])**2 + (obstacle_df['predict_y'] - obstacle_df['oy'])**2
        #obstacle_df = obstacle_df.groupby(['predict_id']).min()
        #print('df len: ', df.shape[0])
        #print(obstacle_df.shape[0])
        #df['cost_obstacle'] = (obstacle_df['obstacle_dist_square'] <= self.config.robot_radius * 1.5) * float('inf')
        df['cost_obstacle'] = 1
        df.loc[self.is_hit_obstacle(df['predict_x'], df['predict_y']), 'cost_obstacle'] = float('inf')

        #df.drop(['tmp'], axis=1, inplace=True)

        # compute total cost
        #df['cost'] = df['cost_velocity'] + df['cost_obstacle'] + df['cost_heading']
        df['cost'] = df['cost_distance'] + df['cost_velocity'] + df['cost_obstacle'] #+ df['cost_heading']


        return df

    def _calc_final_input(self, rover, dynamic_window, goal):
        """ Compute the target velocity, target yawrate, and predicted trajectory dataframe the rover need to set to
        :param rover: a Rover Class `<Rover>`
        :param dynamic_window: a tuple in the form `(v_min, v_max, yawrate_min, yawrate_max)`
        :param goal: a tuple in the form `(x, y)`
        :rtype: A tuple in the form `(target velocity, target yawrate, best trajectory df, all trajectory df)`
        """

        # construct a dataframe for the dynamic window
        df = self._create_dw_dataframe(rover, dynamic_window)

        # compute the overall cost 
        df = self._calc_possible_input_costs(df, rover, goal)
        
        # find the target velocity, yawrate, and best trajectory from the minimum cost
        self.output_df(df)
        print('printed')

        cost_df = df.groupby(['u_velocity', 'u_yawrate']).mean().reset_index()
        u_velocity = cost_df.loc[cost_df['cost'].idxmin()]['u_velocity']
        u_yawrate = cost_df.loc[cost_df['cost'].idxmin()]['u_yawrate']
        best_trajectory_df = df.loc[(df['u_velocity'] == u_velocity) & (df['u_yawrate'] == u_yawrate)].reset_index()

        return u_velocity, u_yawrate, best_trajectory_df, df

    def get_dwa_path(self, rover, goal):
        """ Compute the target velocity, target yawrate, and the predict trajectory of a rover
        :param rover: a Rover Class `<Rover>`
        :param goal: a tuple of floats in the form `(x, y)` [unit in meters]
        :rtype: A tuple in the form `(target velocity, target yawrate, best trajectory df)`
        """

        # compute the dynamic window (the next available velocity and yawrate ranges)
        dynamic_window = self._calc_dynamic_window(rover)

        # compute the target velocity, target yawrate, and the predict trajectory of the rover
        return self._calc_final_input(rover, dynamic_window, goal)

    def output_df(self, df):
        df.to_csv('test.csv', index=False)


def plot_arrow(x, y, yaw, length=250, width=10):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def main():

    config = DWAConfig(control_file='config.json')
    rover = Rover(x=500, y=100, yaw=270.0, v=0, omega=0)
    dwa = DWA(config=config)
    
    gx = 500
    gy = 500
    goal = (gx, gy)

    #dwa.set_obstacles(np.zeros(100) + 300, np.arange(200, 300))


    # dynamic_window = dwa._calc_dynamic_window(rover)
    # print('-- min max velocity')
    # print(dynamic_window[0])
    # print(dynamic_window[1])
    # print('-- min max yawrate')
    # print(dynamic_window[2] * 180 / math.pi)
    # print(dynamic_window[3] * 180 / math.pi)
    
    # df = dwa._create_dw_dataframe(rover, dynamic_window)
    # dwa.output_df(df)

    # plt.cla()
    # plt.plot(df['predict_x'], df['predict_y'], 'ro')
    # plt.pause(0.0001)
    # input()

    for i in range(1000):

        with TaskTimer("Algorithm"):
            u_velocity, u_yawrate, traj_df, all_poss = dwa.get_dwa_path(rover, goal)
        
        print(traj_df['predict_x'][0])
        
        rover.motion(u_velocity, u_yawrate, config.dt)

        # rover.x = traj_df['predict_x'][0]
        # rover.y = traj_df['predict_y'][0]
        # rover.v = traj_df['u_velocity'][0]
        # rover.omega = traj_df['u_yawrate'][0]
        # rover.yaw = traj_df['predict_yaw'][0]

        if True:
            plt.cla()
            plt.plot(np.rot90(dwa.get_obstacles())[1], np.rot90(dwa.get_obstacles())[0], 'o', color='black')
            plt.plot(all_poss['predict_x'], all_poss['predict_y'], '-y')
            plt.plot(traj_df['predict_x'], traj_df['predict_y'], '-g')
            plt.plot(rover.x, rover.y, 'xr')
            plt.plot(goal[0], goal[1], 'xb')
            plot_arrow(rover.x, rover.y, rover.yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        print(rover)
        #input()

        if (rover.x - gx)**2 + (rover.y - gy)**2 <= config.robot_radius**2:
            print("GOAL!")
            input("Press Enter to Exit")
            break
    else:
        print("Failed getting to the goal within the given")

if __name__=="__main__":
    main()
