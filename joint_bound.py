import cdd
import numpy as np
from itertools import repeat
from scipy.spatial import ConvexHull


def partition(bounds, depth=0):
    '''
    :param bounds: the H-representation of the polyhedron before relu
    :param depth: the number of variables to split
    :return:
    '''
    var_num = bounds.shape[1] - 1
    if depth == var_num:
        mat = cdd.Matrix(bounds, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        region = cdd.Polyhedron(mat)
        ext = region.get_generators()
        points_group = np.array(ext)
    else:
        target_var = depth + 1
        new_constrain = np.zeros((1, var_num + 1))
        new_constrain[0, target_var] = 1
        pos_group = partition(np.concatenate([bounds.copy(), new_constrain]), depth + 1)  # positive part
        neg_group = partition(np.concatenate([bounds.copy(), -1 * new_constrain]), depth + 1)  # negative part

        if neg_group.size == 0:
            points_group = pos_group
        elif pos_group.size == 0:
            points_group = neg_group
        else:
            points_group = np.concatenate([pos_group, neg_group])

    return points_group


def points_activate(coefficients, points, activation_fun='relu'):
    """
    The first column is 1, which means its a generator rather than ray in cdd polyhedron.
    :param coefficients:
    :param points:
    :param activation_fun:
    :return:
    """
    assert (points[:, 0] == 1).all(), 'not a bounded area'
    assert coefficients.shape[0] == points.shape[1] - 1, 'the number of dimensions are not same, {} and {}'.format(
        coefficients.shape[0], points.shape[1] - 1
    )

    if activation_fun == 'relu':
        after_act = np.maximum(points[:, 1:], np.zeros_like(points[:, 1:]))
        output = np.dot(after_act, coefficients)
        points = np.concatenate([points, output.reshape((-1, 1))], axis=1)
    else:
        raise NotImplementedError('not implemented for this activation function {}.'.format(activation_fun))

    return points


def points_activate_relu_convex_hull(coefficients, points):
    """
    Points here are pure points, different from function 'points_activated'
    """
    assert coefficients.shape[0] == points.shape[1], 'the number of dimensions are not same, {} and {}'.format(
        coefficients.shape[0], points.shape[1]
    )
    after_act = np.maximum(points, np.zeros_like(points))
    output = np.dot(after_act, coefficients)
    points = np.concatenate([points, output.reshape((-1, 1))], axis=1)

    return points


def cross_constrain(activated_points):
    mat = cdd.Matrix(activated_points, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    mat.canonicalize()

    try:
        # print('good')
        poly = cdd.Polyhedron(mat)
    except:
        # print('rounded number')
        ar = np.around(activated_points, decimals=3)
        # ar = activated_points
        mat = cdd.Matrix(ar, number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
        mat.canonicalize()

        poly = cdd.Polyhedron(mat)

    cons = poly.get_inequalities()
    np_cons = np.array(cons)

    return -1 * np_cons[:, 1:], np_cons[:, 0]


def get_cross_constrain(bounds, coefficient):
    """
    the last column is the output variable, other columns are input variables
    :param bounds: the bounds are [b -A] if the constrains are Ax <= b
    :param coefficient:
    :return:
    """
    unactivated_group = partition(bounds)
    activated_group = points_activate(coefficient, unactivated_group)
    A, b = cross_constrain(activated_group)

    upper_bounds_idx = np.where(A[:, -1] > 0)
    lower_bounds_idx = np.where(A[:, -1] < 0)

    upper_A = A[upper_bounds_idx]
    upper_b = b[upper_bounds_idx]
    lower_A = A[lower_bounds_idx]
    lower_b = b[lower_bounds_idx]

    upper_coef = upper_A[:, -1]
    upper_A = -1 * upper_A[:, :-1] / upper_coef.reshape(-1, 1)
    upper_b = upper_b / upper_coef

    lower_coef = -1 * lower_A[:, -1]
    lower_A = lower_A[:, :-1] / lower_coef.reshape(-1, 1)
    lower_b = -1 * lower_b / lower_coef

    return upper_A, upper_b, lower_A, lower_b


def get_joint_bounds(bounds_A, bounds_b, coefficients):
    """
    If all neurons are activated ,this method is unusable.
    :param bounds_A:
    :param bounds_b:
    :param coefficients:
    :return:
    """
    bounds = np.concatenate([bounds_b.reshape(-1, 1), -1 * bounds_A], axis=1)
    # upper_A, upper_b, lower_A, lower_b = get_cross_constrain(bounds, coefficients)

    unactivated_group = partition(bounds)
    # activated_group = points_activate(coefficients, unactivated_group)
    # A, b = cross_constrain(activated_group)
    # activated_group = activated_group[:, 1:]
    activated_group = points_activate_relu_convex_hull(coefficients, unactivated_group[:, 1:])
    activated_group = np.unique(activated_group, axis=0)
    A, b = convex_hull(activated_group)

    upper_bounds_idx = np.where(A[:, -1] > 0)
    lower_bounds_idx = np.where(A[:, -1] < 0)

    upper_A = A[upper_bounds_idx]
    upper_b = b[upper_bounds_idx]
    lower_A = A[lower_bounds_idx]
    lower_b = b[lower_bounds_idx]

    upper_coef = upper_A[:, -1]
    upper_A = -1 * upper_A[:, :-1] / upper_coef.reshape(-1, 1)
    upper_b = upper_b / upper_coef

    lower_coef = -1 * lower_A[:, -1]
    # change from 'less equal' to 'greater equal'
    lower_A = lower_A[:, :-1] / lower_coef.reshape(-1, 1)
    lower_b = -1 * lower_b / lower_coef

    sample = np.unique(activated_group, axis=0)
    up_A, up_b, low_A, low_b = bound_choice_mc(upper_A, upper_b, lower_A, lower_b, sample)

    return up_A, up_b, low_A, low_b


def bound_choice_intercept(upper_A, upper_b, lower_A, lower_b):
    idx_up = np.argmin(np.abs(upper_b))
    up_A = upper_A[idx_up]
    up_b = upper_b[idx_up]
    idx_low = np.argmin(np.abs(lower_b))
    low_A = lower_A[idx_low]
    low_b = lower_b[idx_low]

    return up_A, up_b, low_A, low_b


def bound_choice_mc(upper_A, upper_b, lower_A, lower_b, samples):
    '''
    Use monte carlo method to choose bounds.
    :param upper_A:
    :param upper_b:
    :param lower_A:
    :param lower_b:
    :param samples:
    :return:
    '''
    actual_value = samples[:, -1]
    samples = samples[:, :-1]

    upper_A = upper_A[:, np.newaxis, :]
    up_diff = np.matmul(upper_A, np.transpose(samples))
    up_diff = np.squeeze(up_diff)
    up_diff = up_diff + upper_b.reshape(-1, 1) - actual_value.reshape(1, -1)
    up_res = np.sum(np.abs(up_diff), axis=1)
    up_idx = np.argmin(up_res)
    up_A = upper_A[up_idx]
    up_b = upper_b[up_idx]

    lower_A = lower_A[:, np.newaxis, :]
    low_diff = np.matmul(lower_A, np.transpose(samples))
    low_diff = np.squeeze(low_diff)
    low_diff = low_diff + lower_b.reshape(-1, 1) - actual_value.reshape(1, -1)
    low_res = np.sum(np.abs(low_diff), axis=1)
    low_idx = np.argmin(low_res)
    low_A = lower_A[low_idx]
    low_b = lower_b[low_idx]

    return up_A, up_b, low_A, low_b


def load_poly(points_set):
    mat = cdd.Matrix(points_set)
    mat.rep_type = cdd.RepType.GENERATOR

    try:
        # print('good')
        poly = cdd.Polyhedron(mat)
    except:
        # print('rounded number')
        ar = np.around(points_set, decimals=3)
        # ar = activated_points
        mat = cdd.Matrix(ar, number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
        mat.canonicalize()

        poly = cdd.Polyhedron(mat)

    return poly


def convex_hull(points):
    """
    The return format of 'equations' is [V, b], which means Vx+b <= 0.
    :param points:
    :return:
    """
    hull = ConvexHull(points)
    facet = hull.equations

    return facet[:, :-1], -1 * facet[:, -1]


if __name__ == '__main__':
    A = np.array([[ 0.,  1.], [ 0., -1.], [ 1.,  0.], [ 1.,  1.], [ 1., -1.], [-1.,  0.], [-1.,  1.], [-1., -1.]])
    # coeff = np.array([0.07256681, 0.0658548])
    # b = np.array([3, 1, 1, 3, 1, 1, 3, 1])
    b = np.array([2, 1, 1, 2, 1, 1, 2, 1])
    # b = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    coeff = np.array([1, 0.5])
    uA, ub, _, _ = get_joint_bounds(A, b, coeff)