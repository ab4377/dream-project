import pandas as pd
import numpy as np


def circular_range(start, stop, step_size):
    """
    Returns a range that wraps around 2*pi, i.e
    if start = pi, stop = pi/2, then the range will wraps back to 0 once it gets past
    2*pi
    :param start: Start angle
    :param stop: Stop angle (must be smaller than 2*pi)
    :param step_size: Step in angle
    :return: Generator that generates the range specified.
    """
    step = 0
    r = start
    if stop < start:
        stop += (2 * np.pi)
    angles = np.arange(start, stop, step_size) % (2 * np.pi)
    return angles


def rotate_3d(vect, C_matrix, theta, axis):
    R_matrix = rotate_3d_matrix(theta, axis)
    C_inv_matrix = np.linalg.inv(C_matrix)
    return np.array(np.linalg.multi_dot([C_matrix, R_matrix, C_inv_matrix, vect[0]]))


def rotate_3d_matrix(theta, axis):
    """
    Returns a 3d matrix given the rotation angle and the axis of rotation.
    :param theta: The rotation angle
    :param axis: Axis of rotation: 0 = x-axis, 1 = y-axis, 2 = z-axis
    :return: The 3 x 3 rotation matrix for the given angle and axis.
    """

    if axis == 0:
        rotation_matrix = np.matrix([[1, 0,             0             ],
                                     [0, np.cos(theta), -np.sin(theta)],
                                     [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        rotation_matrix = np.matrix([[np.cos(theta), 0, np.sin(theta)],
                                     [0,             1,             0],
                                     [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 2:
        rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta),  0],
                                     [0,             0,              1]])
    else:
        raise ValueError('axis is ' + axis + ', while expected to be either 0, 1, or 2')
    return rotation_matrix


def project(a, b):
    """
    Returns the project of vector a on vector b.
    :param a: The vector being projected.
    :param b: The vector being projected on.
    :return: Scalar representing the projection of a on b.
    """
    return (np.dot(a, b.transpose()) / np.linalg.norm(b[0]))[0, 0]


def normalize_vect(vector):
    return vector / np.linalg.norm(vector)


def project_array(projecting_vectors, projected_vector):
    """
    Project all the vectors in an array of vectors onto a single vector (returning a list of scalars)
    :param projecting_vectors: array of vectors
    :param projected_vector: vector being projected on
    :return: array of scalars, each is a projection of a vector in the project_vectors array onto vector.
    """
    norm_vector = np.linalg.norm(projected_vector)
    dot_products = np.dot(projecting_vectors, np.transpose(projected_vector))
    return dot_products / norm_vector


def find_longest_positive_stretch(arr):
    positive_values_indices = np.where(arr >= 0)[0]
    positive_stretches = np.split(positive_values_indices, np.where(np.diff(positive_values_indices) != 1)[0] + 1)
    longest_positive_stretch = np.argmax(np.array([stretch.size for stretch in positive_stretches]))
    return positive_stretches[longest_positive_stretch]
