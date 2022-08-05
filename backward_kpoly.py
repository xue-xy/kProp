import torch
import numpy as np
from itertools import product
import copy

import util
from joint_bound import get_joint_bounds
import backward_init as bdp


def back_prop(target_A, target_b, weight, bias, relations, norm, data0, radius, group_num, verbose=False):
    '''
    :param target_A: target weights after activation layer to be processed
    :param target_b:
    :param weight: the weight and bias before the activation layer to be processed
    :param bias:
    :param relations: the relations before the weight
    :param norm:
    :param data0:
    :param radius:
    :param group_num:
    :return:
    '''
    pos_weight = target_A > 0

    up_A, up_b, low_A, low_b = bdp.back_prop(weight, bias, relations)
    ub, lb = util.numerical_bound(up_A, up_b, low_A, low_b, norm, data0, radius)

    nonzero = (ub >= 0)

    pos_nonzero = torch.logical_and(pos_weight, nonzero)
    neg_nonzero = torch.logical_and(torch.logical_not(pos_weight), nonzero)

    pos_nonzero_idx = torch.where(pos_nonzero == True)[0].numpy()
    neg_nonzero_idx = torch.where(neg_nonzero == True)[0].numpy()

    target_Ap = torch.zeros_like(target_A)

    bounds_A = get_constrain_expression(group_num)
    gp, gn = 0, 0
    while gp < pos_nonzero_idx.shape[0] // group_num:
        idx = pos_nonzero_idx[gp*group_num: (gp+1)*group_num]
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)

        target_b += Ub
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)
        gp += 1

    while gn < neg_nonzero_idx.shape[0] // group_num:
        idx = neg_nonzero_idx[gn*group_num: (gn+1)*group_num]
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)

        target_b += Ub
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)
        gn += 1

    # process the remaining nonzero part
    if pos_nonzero_idx.shape[0] % group_num != 0:
        idx = pos_nonzero_idx[gp * group_num:]
        bounds_A = get_constrain_expression(idx.shape[0])
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)
        target_b += Ub
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)

    if neg_nonzero_idx.shape[0] % group_num:
        idx = neg_nonzero_idx[gn * group_num:]
        bounds_A = get_constrain_expression(idx.shape[0])
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)
        target_b += Ub
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)

    target_A = torch.matmul(target_Ap, weight)
    target_b = torch.matmul(target_Ap, bias) + target_b

    return target_A, target_b


def back_prop_und(target_A, target_b, weight, bias, relations, norm, data0, radius, group_num, verbose=False):
    '''
    only deal with the undetermined neurons
    :param target_A:
    :param target_b:
    :param weight:
    :param bias:
    :param relations:
    :param norm:
    :param data0:
    :param radius:
    :param group_num:
    :param verbose:
    :return:
    '''

    up_A, up_b, low_A, low_b = bdp.back_prop(weight, bias, relations)
    ub, lb = util.numerical_bound(up_A, up_b, low_A, low_b, norm, data0, radius)
    und = torch.where(torch.logical_and(lb < 0, ub > 0))[0].numpy()

    target_Ap = torch.zeros_like(target_A)
    pos = torch.where(lb >= 0)
    target_Ap[pos] = target_A[pos]

    bounds_A = get_constrain_expression(group_num)
    gi = 0
    while gi < und.shape[0] // group_num:
        idx = und[gi * group_num: (gi + 1) * group_num]
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)
        target_b += Ub
        gi += 1

    # process the remaining part
    if und.shape[0] % group_num != 0:
        idx = und[gi * group_num:]
        bounds_A = get_constrain_expression(idx.shape[0])
        UA, Ub = idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub)
        target_Ap[idx] = torch.tensor(UA, dtype=torch.float)
        target_b += Ub

    target_A = torch.matmul(target_Ap, weight)
    target_b = torch.matmul(target_Ap, bias) + target_b
    return target_A, target_b


def get_constrain_expression(neuron_group_num):
    bounds_iter = product([0, 1, -1], repeat=neuron_group_num)
    next(bounds_iter) # throw out the all zero expression

    bounds_A = np.zeros((0, neuron_group_num))
    for i in bounds_iter:
        bounds_A = np.concatenate([bounds_A, np.array([i])])

    return bounds_A


def idx2bounds(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub):
    coefficient = target_A[idx].numpy()
    if (lb[idx] >= 0).all():
        return coefficient, 0
    # if np.sum(np.abs(coefficient)) == 0:
    #     return coefficient, 0
    expr_A, expr_b, _, _ = util.bounds_prop(torch.tensor(bounds_A, dtype=torch.float),
                                            torch.zeros(bounds_A.shape[0]),
                                            up_A[idx], up_b[idx], low_A[idx], low_b[idx])
    bounds_b = util.numerical_upper_bound(expr_A, expr_b, norm, data0, radius).numpy()

    try:
        UA, Ub, LA, Lb = get_joint_bounds(bounds_A, bounds_b, coefficient)
    except:
        UA = np.zeros(idx.shape[0])
        Ub = 0
        for i in range(idx.shape[0]):
            if lb[idx[i]] >= 0:
                UA[i] = target_A[idx[i]]
            else:
                if coefficient[i] >= 0:
                    UA[i] = target_A[idx[i]] * ub[idx[i]] / (ub[idx[i]] - lb[idx[i]])
                    Ub += target_A[idx[i]] * -1 * lb[idx[i]] * ub[idx[i]] / (ub[idx[i]] - lb[idx[i]])
                else:
                    UA[i] = target_A[idx[i]] * ub[idx[i]] / (ub[idx[i]] - lb[idx[i]])

    return UA, Ub


def idx2bound_und(idx, target_A, bounds_A, up_A, up_b, low_A, low_b, norm, data0, radius, lb, ub):
    pass


def kprop_framework(weights, biases, testdata, testlabel, norm, radius, group_num, verbose=False):
    label = testlabel.item()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    testdata = testdata.to(device)

    relations = ([], [], [], [])

    for layer in range(len(weights) - 1):
        weight = weights[layer].to(device)
        bias = biases[layer].to(device)
        relations = bdp.layer_relation(weight, bias, relations, norm, testdata, radius)

    # do deeppoly here
    dres = np.zeros(10) - 1
    for i in range(10):
        if i == label:
            continue
        target_A = (weights[-1][i] - weights[-1][label]).to(device)
        target_b = (biases[-1][i] - biases[-1][label]).to(device)
        dUA, dUb, dLA, dLb = bdp.back_prop(target_A, target_b, relations)

        ub0 = util.numerical_upper_bound(dUA, dUb, norm, testdata, radius)
        dres[i] = ub0.item()
    if (dres < 0).all():
        if verbose:
            print('verified')
        return True
    relations = (relations[0][:-1], relations[1][:-1], relations[2][:-1], relations[3][:-1])
    # if deeppoly does not work, do k poly

    res = np.zeros(10) - 1
    for i in range(10):
        if i == label:
            continue
        target_A = (weights[-1][i] - weights[-1][label]).to(device)
        target_b = (biases[-1][i] - biases[-1][label]).to(device)

        if verbose:
            print(i)

        subrelation = copy.deepcopy(relations)
        layer = len(weights) - 2
        while layer > 0:
            target_A, target_b = back_prop_und(target_A, target_b, weights[layer], biases[layer], subrelation,
                                           norm, testdata, radius, group_num)
            subrelation = (subrelation[0][:-1], subrelation[1][:-1], subrelation[2][:-1], subrelation[3][:-1])
            layer = layer - 1

            if verbose:
                print(layer)

        target_A, target_b = back_prop_und(target_A, target_b, weights[layer], biases[layer], subrelation,
                                       norm, testdata, radius, group_num)

        if norm == 'inf':
            ub0 = torch.linalg.norm(target_A, ord=1) * radius + torch.matmul(target_A, testdata) + target_b
        elif norm == '1':
            ub0 = torch.linalg.norm(target_A, ord=float('inf')) * radius + torch.matmul(target_A, testdata) + target_b
        elif norm == '2':
            ub0 = torch.linalg.norm(target_A, ord=2) * radius + torch.matmul(target_A, testdata) + target_b
        res[i] = ub0.item()

    if verbose:
        print('res:', res)
    if (res < 0).all():
        if verbose:
            print('verified')
        return True
    else:
        if verbose:
            print('not verified', np.max(res))
        return False


def kprop_framework_low(weights, biases, testdata, testlabel, norm, radius, group_num, verbose=False):
    #todo: change this to low
    label = testlabel.item()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    testdata = testdata.to(device)

    relations = ([], [], [], [])

    for layer in range(len(weights) - 1):
        weight = weights[layer].to(device)
        bias = biases[layer].to(device)
        relations = bdp.layer_relation(weight, bias, relations, norm, testdata, radius)

    # do deeppoly here
    dres = np.zeros(10) - 1
    for i in range(10):
        if i == label:
            continue
        target_A = (weights[-1][i] - weights[-1][label]).to(device)
        target_b = (biases[-1][i] - biases[-1][label]).to(device)
        dUA, dUb, dLA, dLb = bdp.back_prop(target_A, target_b, relations)

        if norm == 'inf':
            ub0 = torch.linalg.norm(dUA, ord=1) * radius + torch.matmul(dUA, testdata) + dUb
        elif norm == '1':
            ub0 = torch.linalg.norm(dUA, ord=float('inf')) * radius + torch.matmul(dUA, testdata) + dUb
        elif norm == '2':
            ub0 = torch.linalg.norm(dUA, ord=2) * radius + torch.matmul(dUA, testdata) + dUb
        dres[i] = ub0.item()
    if (dres < 0).all():
        if verbose:
            print('verified')
        return True
    relations = (relations[0][:-1], relations[1][:-1], relations[2][:-1], relations[3][:-1])
    # if deeppoly does not work, do k poly

    res = np.zeros(10) - 1
    for i in range(10):
        if i == label:
            continue
        target_A = (weights[-1][i] - weights[-1][label]).to(device)
        target_b = (biases[-1][i] - biases[-1][label]).to(device)

        subrelation = copy.deepcopy(relations)
        layer = len(weights) - 2
        while layer > 0:
            target_A, target_b = back_prop_und(target_A, target_b, weights[layer], biases[layer], subrelation,
                                           norm, testdata, radius, group_num)
            subrelation = (subrelation[0][:-1], subrelation[1][:-1], subrelation[2][:-1], subrelation[3][:-1])
            layer = layer - 1

        target_A, target_b = back_prop_und(target_A, target_b, weights[layer], biases[layer], subrelation,
                                       norm, testdata, radius, group_num)

        if norm == 'inf':
            ub0 = torch.linalg.norm(target_A, ord=1) * radius + torch.matmul(target_A, testdata) + target_b
        elif norm == '1':
            ub0 = torch.linalg.norm(target_A, ord=float('inf')) * radius + torch.matmul(target_A, testdata) + target_b
        elif norm == '2':
            ub0 = torch.linalg.norm(target_A, ord=2) * radius + torch.matmul(target_A, testdata) + target_b
        res[i] = ub0.item()

    if verbose:
        print('res:', res)
    if (res < 0).all():
        if verbose:
            print('verified')
        return True
    else:
        if verbose:
            print('not verified', np.max(res))
        return False



