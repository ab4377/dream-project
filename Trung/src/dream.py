import pandas as pd
import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import math
import json
import pyhsmm
import pyhsmm.basic.distributions as distributions
from pyhsmm.util.text import progprint_xrange


#######################################################################################################################
def fit_hmm(data, max_num_states=25, max_duration=100, num_iter=200, obs_dist=None, dur_dist=None,
            print_prog=False):

    obs_dim = data.shape[1]

    obs_hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 0.3,
                     'nu_0': obs_dim + 5}
    dur_hypparams = {'alpha_0': 2*30,
                     'beta_0': 2}

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(max_num_states)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(max_num_states)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6., gamma=6.,
        init_state_concentration=4.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

    posteriormodel.add_data(data, trunc=max_duration)

    if print_prog:
        iter_range = progprint_xrange(num_iter)
    else:
        iter_range = xrange(num_iter)

    for idx in iter_range:
        posteriormodel.resample_model()

    return posteriormodel


def initialize_distns(max_num_states, obs_dim):
    hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 0.3,
                     'nu_0': obs_dim + 5}

    distns = [distributions.Gaussian(**hypparams) for state in range(max_num_states)]

    return distns


def extract_unique_seqs(seq):
    copy_seq = np.copy(seq)
    states = np.unique(copy_seq)
    [np.place(copy_seq, seq == states[i], i) for i in range(len(states))]
    return copy_seq


def count_states(seq):
    # note, the state marker has to be in consecutive value, e.g 0, 1, 2, 3, not
    # 2, 16, 21
    counts = np.array([np.sum(np.where(seq == i, [1], [0])) for i in range(len(np.unique(seq)))])
    return counts


def find_stride_durations(seq, dur):
    counts = count_states(seq)
    med = np.median(counts)
    repeating_state = np.argmin(np.absolute(counts - med))
    start = np.where(seq == repeating_state)[1]
    stride_durations = [np.sum(dur[start[i-1]:start[i]:1]) for i in range(1, len(start))]

    return stride_durations


def get_slices_for_state(seq, dur, state, data):
    slice_starts = np.where(seq == state)[1]
    slices = [data[slice_start:slice_start + dur[i]:1] ]

#######################################################################################################################
def find_forward_backward_direction(motion_df, target_delta_theta, visualize = False, visualize_path =""):
    """
    When given a time series of acceleration and gravity vectors, this function finds the backward-forward
    direction by trying different directions in the plane orthogonal to the gravity vector. The direction that
    maximizes the distance traveled in the forward-backward direction when chosen as the forward-backward direction will
    be chosen as the forward-backward direction. In short, this algorithm assumes that most of the distance traveled
    comes from the forward-backward direction and tries to maximize that distance.

    :param motion_df: A Pandas data frame containing a time series of acceleration and gravity vectors.
    :param target_delta_theta: The target precision of the rotation angle (the true theta angle that maximizes distance
    will be within target_delta_theta of the theta angle found by the algorithm).
    :param visualize: Whether the resulting time series of velocities will be visualized in a graph.
    :param visualize_path: The path where the graphs will be saved. If empty, these graphs will be shown interactively
    :return: A tuple (max_fb_vect, max_theta, max_distance), where max_fb_vect is the backward-forward direction found
    by the algorithm, max_theta is rotation angle of the backward-forward direction found compared to the initial
    backward-forward direction, and max_distance is the distance traveled in the backward-forward direction found.
    """
    init_fb_vect, init_side_vect, C_matrix, acceleration_vects, times = init_vars(motion_df)
    (max_distance, max_fb_vect, max_theta, delta_theta, start_theta, end_theta) = (0, init_fb_vect, 0, np.pi / 3, 0, 2 * np.pi)

    while delta_theta > target_delta_theta:
        thetas = utils.circular_range(start_theta + delta_theta, end_theta, delta_theta)
        fb_vects = [utils.rotate_3d(init_fb_vect, C_matrix, theta, 2) for theta in thetas]
        distances, velocities = zip(*(calculate_distance_for_fb_direction(acceleration_vects, times, fb_vect)
                                      for fb_vect in fb_vects))
        max_ind, start_theta, end_theta, delta_theta = recalculate_theta(distances, thetas, delta_theta)
        max_fb_vect, max_theta, max_distance = (fb_vects[max_ind], thetas[max_ind], distances[max_ind])
        if visualize:
            visualize_velocities(times, velocities[max_ind], thetas[max_ind], visualize_path)

    return max_fb_vect, max_theta, max_distance


def init_vars(motion_df):
    # Initialize forward-backward direction by picking a random direction in
    # the plane perpendicular to the z-axis (indicated by the gravity vector).
    # For each rotation of theta radians of the initial direction, find
    # longest sequence of positive velocity and calculate distance.
    gravity_vect = np.array([[motion_df.loc[0, dim] for dim in ['gx', 'gy', 'gz']]])
    (x_g, y_g, z_g) = tuple(gravity_vect[0, dim] for dim in [0, 1, 2])
    init_fb_vect = utils.normalize_vect(np.array([[y_g / x_g, -1, 0]]))
    init_side_vect = utils.normalize_vect(np.array([[x_g, y_g, -(x_g ** 2 + y_g ** 2) / z_g]]))
    C_matrix = np.column_stack((init_side_vect[0], init_fb_vect[0], gravity_vect[0]))
    acceleration_vects = np.array([[motion_df.loc[row, dim] for dim in ['x', 'y', 'z']]
                                   for row in range(1, motion_df.shape[0])])
    times = np.diff(np.array(motion_df['timestamp']))
    return init_fb_vect, init_side_vect, C_matrix, acceleration_vects, times


def calculate_distance_for_fb_direction(acceleration_vects, times, fb_vect):
    forward_acceleration_vects = utils.project_array(acceleration_vects, fb_vect)
    velocities = np.cumsum(forward_acceleration_vects[:, 0] * times)
    longest_positive_stretch = utils.find_longest_positive_stretch(velocities)
    if longest_positive_stretch.size == 0:
        return 0, velocities
    distance = np.cumsum(velocities[longest_positive_stretch] * times[longest_positive_stretch])[-1]
    return distance, velocities



def recalculate_theta(distances, thetas, old_delta_theta):
    # Take the theta that maximizes distance and search for
    # a better resolution by looking at the range
    # (max_theta - old_delta_theta, max_theta + old_delta_theta)
    max_ind = np.argmax(distances)
    new_start_theta = thetas[max_ind] - old_delta_theta
    new_end_theta = thetas[max_ind] + old_delta_theta
    new_delta_theta = (new_end_theta - new_start_theta) / 6
    return max_ind, new_start_theta, new_end_theta, new_delta_theta


def visualize_velocities(time_stamps, velocities_fb, theta, visualize_path=""):
    time_stamps = time_stamps.tolist()
    time_stamps.pop(0)
    fig, ax = plt.subplots()
    ax.plot(time_stamps, velocities_fb, label='Forward-backward')
    ax.legend(loc='lower right')

    if visualize_path == "":
        plt.show()
    else:
        if os.path.exists(visualize_path):
            plt.savefig(visualize_path + '/' + str(theta) + '.png')
            plt.close()
        else:
            raise IOError('visualize_path directory does not exist')


#######################################################################################################################

def relative_to_absolute_coordinates_file(input_file, output_file):
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
    results = np.empty(0)
    for item in data:
        result = relative_to_absolute_coordinates_item(item)
        results = result if results.size == 0 else np.vstack((results, result))

    results = pd.DataFrame(results, columns=['timestamp', 'x', 'y', 'z', 'gx', 'gy', 'gz'])
    results.to_csv(output_file)


def relative_to_absolute_coordinates_item(item):
    a_vector = np.array([item.get('userAcceleration').get(a) for a in ['x', 'y', 'z']])
    g_vector = np.array([item.get('attitude').get(g) for g in ['x', 'y', 'z']])
    (a, b, c, d) = tuple(item.get('attitude').get(att) for att in ['x', 'y', 'z', 'w'])
    rotation_matrix = calculate_rotation_matrix(a, b, c, d)

    abs_a = np.dot(rotation_matrix, a_vector)
    abs_g = np.dot(rotation_matrix, g_vector)

    time_stamp = np.array([item.get('timestamp')])
    return np.concatenate((time_stamp, abs_a, abs_g))


def calculate_rotation_matrix(a, b, c, d):
    return np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                     [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                     [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]])

#######################################################################################################################

def convert_to_forward_backward_coordinates(df, fb_vect):
    gravity_vect = np.array(df.loc[0, ['gx', 'gy', 'gz']].tolist())
    sideway_vect = np.cross(fb_vect, gravity_vect)

    fb_acc = utils.project_array(df[['x', 'y', 'z']], fb_vect)
    updown_acc = utils.project_array(df[['x', 'y', 'z']], gravity_vect)
    sideway_acc = utils.project_array(df[['x', 'y', 'z']], sideway_vect)
    d = {'timestamp': df['timestamp'], 'x': fb_acc, 'y': sideway_acc, 'z': updown_acc}
    return pd.DataFrame(d)