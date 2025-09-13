#######################################################################
# Name: sensor.py
#
# - Computes sensor related checks (e.g. collision, utility etc)
#######################################################################

import sys
if sys.modules['TRAINING']:
    from .parameter import *
else:
    from .test_parameter import *

import math
import numpy as np
import copy

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 10

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k !=1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth, sensor_model=SENSOR_MODEL):
    x0 = robot_position[0]
    y0 = robot_position[1]
    rng_x = 0.5 * (ground_truth.shape[1] / NUM_COORDS_WIDTH)
    rng_y = 0.5 * (ground_truth.shape[0] / NUM_COORDS_HEIGHT)

    if sensor_model == "rectangular":   # TODO: add collision check 
        max_x = min(x0 + int(math.ceil(rng_x)), ground_truth.shape[1])
        min_x = max(x0 - int(math.ceil(rng_x)), 0)
        max_y = min(y0 + int(math.ceil(rng_y)), ground_truth.shape[0])
        min_y = max(y0 - int(math.ceil(rng_y)), 0)
        robot_belief[min_y:max_y, min_x:max_x] = ground_truth[min_y:max_y, min_x:max_x]
    else:
        sensor_angle_inc = 0.5 / 180 * np.pi
        sensor_angle = 0
        while sensor_angle < 2 * np.pi:
            x1 = x0 + np.cos(sensor_angle) * sensor_range
            y1 = y0 + np.sin(sensor_angle) * sensor_range
            robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
            sensor_angle += sensor_angle_inc
    return robot_belief


def unexplored_area_check(x0, y0, x1, y1, current_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < current_belief.shape[1] and 0 <= y < current_belief.shape[0]:
        k = current_belief.item(y, x)
        if x == x1 and y == y1:
            break

        if k == 1:
            break

        if k == 127:
            current_belief.itemset((y, x), 0)
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return current_belief


def calculate_utility(waypoint_position, sensor_range, robot_belief):
    sensor_angle_inc = 5 / 180 * np.pi
    sensor_angle = 0
    x0 = waypoint_position[0]
    y0 = waypoint_position[1]
    current_belief = copy.deepcopy(robot_belief)
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        current_belief = unexplored_area_check(x0, y0, x1, y1, current_belief)
        sensor_angle += sensor_angle_inc
    utility = np.sum(robot_belief == 127) - np.sum(current_belief == 127)
    return utility
