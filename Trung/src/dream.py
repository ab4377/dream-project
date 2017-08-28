import pandas as pd
import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import math
import json


def find_forward_backward_direction(motion_df, target_delta_theta, visualize = False, visualize_path =""):
    """
    When given a time series of acceleration and gravity vectors, this function finds the backward-forward
    direction by trying different directions in the plane orthogonal to the gravity vector. The direction that
    maximizes the distance traveled in the forward-backward direction when chosen as the forward-backward direction will
    be chosen as the forward-backward direction. In short, this algorithm assumes that most of the distance traveled
    comes from the forward-backward direction and tries to maximize that distance.

    :param motion_df: A Pandas data frame containing a time series of acceleration and gravity vectors.
    :param target_delta_theta: The target resolution of the rotation angle (the true theta angle that maximizes distance
    will be within target_delta_theta of the theta angle found by the algorithm).
    :param visualize: Whether the resulting time series of velocities will be visualized in a graph.
    :param visualize_path: The path where the graphs will be saved. If empty, these graphs will be shown interactively
    :return: A tuple (max_fb_vect, max_theta, max_distance), where max_fb_vect is the backward-forward direction found
    by the algorithm, max_theta is rotation angle of the backward-forward direction found compared to the initial
    backward-forward direction, and max_distance is the distance traveled in the backward-forward direction found.
    """
    # Initialize forward-backward direction by picking a random direction in
    # the plane perpendicular to the z-axis (indicated by the gravity vector).
    # For each rotation of theta radians of the initial direction, find
    # longest sequence of positive velocity and calculate distance.
    x_g = motion_df.loc[0, 'gx']
    y_g = motion_df.loc[0, 'gy']
    z_g = motion_df.loc[0, 'gz']
    downward_vect = np.array([[x_g, y_g, z_g]])
    init_fb_vect = np.array([[y_g / x_g, -1, 0]])
    init_side_vect = np.array([[x_g, y_g, -(x_g ** 2 + y_g ** 2) / z_g]])
    init_fb_vect = init_fb_vect / np.linalg.norm(init_fb_vect)
    init_side_vect = init_side_vect / np.linalg.norm(init_side_vect)
    assert (init_fb_vect.dot(init_side_vect.transpose()) <= 0.0001)
    assert (init_fb_vect.dot(downward_vect.transpose()) <= 0.0001)

    # v_r = C * R * C' * v, where v_r is the rotated vector and v
    # is the original vector in the absolute coordinates.
    # C' maps the original vector from the
    # absolute coordinates to the coordinates formed by the downward,
    # forward-backward, and sideways vector, R rotates the vector in the
    # same space, and C reconverts the vector to the absolute coordinates.
    # Here, x = sideways, y = forward-backward, z = downward
    C_matrix = np.column_stack((init_side_vect[0], init_fb_vect[0], downward_vect[0]))
    C_inv_matrix = np.linalg.inv(C_matrix)
    max_distance = 0
    max_fb_vect = init_fb_vect
    max_theta = 0
    delta_theta = np.pi / 3
    start_theta = 0
    end_theta = 2 * np.pi

    while delta_theta > target_delta_theta:
        for theta in utils.circular_range(start_theta, end_theta, delta_theta):
            R_matrix = utils.rotate_3d_matrix(theta, 2)
            fb_vect = np.array(np.linalg.multi_dot([C_matrix, R_matrix, C_inv_matrix, init_fb_vect[0]]))
            side_vect = np.array(np.linalg.multi_dot([C_matrix, R_matrix, C_inv_matrix, init_side_vect[0]]))
            velocities = []
            velocities_sideway = []
            curr_velocity = 0
            curr_velocity_sideway = 0
            distance = 0
            for row in range(motion_df.shape[0]):
                if (row == 0):
                    continue
                x_a = motion_df.loc[row, 'x']
                y_a = motion_df.loc[row, 'y']
                z_a = motion_df.loc[row, 'z']

                acceleration_vect = np.array([[x_a, y_a, z_a]])
                forward_acceleration = utils.project(acceleration_vect, fb_vect)
                sideway_acceleration = utils.project(acceleration_vect, side_vect)

                time = motion_df.loc[row, 'timestamp'] - motion_df.loc[row - 1, 'timestamp']
                curr_velocity += forward_acceleration * time
                curr_velocity_sideway += sideway_acceleration * time

                velocities.append(curr_velocity)
                velocities_sideway.append(curr_velocity_sideway)

                if curr_velocity > 0:
                    distance += curr_velocity
                else:
                    if distance > max_distance:
                        max_distance = distance
                        max_theta = theta
                        max_fb_vect = fb_vect
                        distance = 0

            if distance > max_distance:
                max_distance = distance
                max_theta = theta
                max_fb_vect = fb_vect

            if visualize:
                time_stamps = motion_df.loc[:, 'timestamp'].tolist()
                time_stamps.pop(0)
                fig, ax = plt.subplots()
                ax.plot(time_stamps, velocities, label='Forward-backward')
                ax.plot(time_stamps, velocities_sideway, label='Sideways')
                ax.legend(loc='lower right')

                if visualize_path == "":
                    plt.show()
                else:
                    if os.path.exists(visualize_path):
                        plt.savefig(visualize_path + '/' + str(theta) + '.png')
                        plt.close()
                    else:
                        raise IOError('visualize_path directory does not exist')
        start_theta = max_theta - delta_theta
        end_theta = max_theta + delta_theta
        delta_theta = (end_theta - start_theta) / 6

    return max_fb_vect, max_theta, max_distance


def relative_to_absolute_coordinates(input_file, output_file):
    """
    Given an input file containing device motion information (time series of acceleration, gravity, attitude, etc., as
    defined in the Synapse data description), outputs a csv file containing the acceleration and gravity in absolute
    coordinates.
    :param input_file: The input file containing device motion information.
    :param output_file: The output file, a csv file containing a time series of acceleration and gravity in absolute
    coordinates.
    :return: No return value
    """
    data = json.load(open(input_file))
    timestamp = []
    results = np.empty(0)
    for item in data:
        a_x = item.get("userAcceleration").get("x")
        a_y = item.get("userAcceleration").get("y")
        a_z = item.get("userAcceleration").get("z")
        a_vector = np.array([a_x, a_y, a_z])

        g_x = item.get("gravity").get("x")
        g_y = item.get("gravity").get("y")
        g_z = item.get("gravity").get("z")
        g_vector = np.array([g_x, g_y, g_z])

        a = item.get("attitude").get("x")
        b = item.get("attitude").get("y")
        c = item.get("attitude").get("z")
        d = item.get("attitude").get("w")
        rotation_matrix = np.matrix([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                                     [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                                     [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]])

        abs_a = np.dot(rotation_matrix, a_vector)
        abs_g = np.dot(rotation_matrix, g_vector)

        abs_a_without_gravity = np.array(abs_a - abs_g)[0]

        timestamp = np.array([item.get("timestamp")])
        row = np.concatenate((timestamp, abs_a_without_gravity, abs_g))

        if (results.size == 0):
            results = row
        else:
            results = np.vstack((results, row))

    results = pd.DataFrame(results, columns=['timestamp', "x", "y", "z", "gx", "gy", "gz"])
    results.to_csv(output_file)
