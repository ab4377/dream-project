import pandas as pd
import numpy as np


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