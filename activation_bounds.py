import torch
import numpy as np


def relu_bounds_simple(U_A, U_b, L_A, L_b, ub, lb):
    """
    A simple version. Use y = 0 for lower bound.
    :param U_A:
    :param U_b:
    :param L_A:
    :param L_b:
    :param lb:
    :param ub:
    :return:
    """
    pos = torch.where(lb >= 0)
    neg = torch.where(ub <= 0)
    und = torch.where(torch.logical_and(lb < 0, ub > 0))

    # print('neg', neg[0].shape[0], 'pos', pos[0].shape[0], 'und', und[0].shape[0])

    U_A[neg] = torch.zeros_like(U_A[neg])
    U_b[neg] = 0
    L_A[neg] = torch.zeros_like(L_A[neg])
    L_b[neg] = 0

    slop = ub[und]/(ub[und] - lb[und])
    U_A[und] = slop.view(-1,1) * U_A[und]
    U_b[und] = slop * U_b[und] - slop * lb[und]
    L_A[und] = torch.zeros_like(L_A[und])
    L_b[und] = 0
    # L_A[und] = slop.view(-1,1) * U_A[und]
    # L_b[und] = 0

    return U_A, U_b, L_A, L_b


def relu_relation_simple(ub, lb):
    neg = torch.where(ub <= 0)
    pos = torch.where(lb >= 0)
    und = torch.where(torch.logical_and(lb < 0, ub > 0))

    # print('neg', neg[0].shape[0], 'pos', pos[0].shape[0], 'und', und[0].shape[0])

    UA = torch.zeros_like(ub)
    Ub = torch.zeros_like(ub)
    LA = torch.zeros_like(ub)
    Lb = torch.zeros_like(ub)

    UA[pos] = 1
    LA[pos] = 1

    slop = ub[und] / (ub[und] - lb[und])
    UA[und] = slop
    Ub[und] = -1 * slop * lb[und]

    neg_und = torch.where(torch.logical_and(torch.logical_and(lb < 0, ub > 0), lb + ub < 0))
    pos_und = torch.where(torch.logical_and(torch.logical_and(lb < 0, ub > 0), lb + ub >= 0))
    LA[neg_und] = 0
    LA[pos_und] = 1
    Lb[und] = 0

    return UA, Ub, LA, Lb


def relu_relation_matrix(ub, lb):
    UA_row = torch.zeros_like(ub)
    Ub = torch.zeros_like(ub)
    LA_row = torch.zeros_like(ub)
    Lb = torch.zeros_like(ub)

    pos = torch.where(lb >= 0)
    UA_row[pos] = 1
    LA_row[pos] = 1

    und_bool = torch.logical_and(lb < 0, ub > 0)
    und = torch.where(und_bool)
    UA_row[und] = ub[und] / (ub[und] - lb[und])
    Ub[und] = -1 * lb[und] * ub[und] / (ub[und] - lb[und])

    p_und = torch.where(torch.logical_and(und_bool, ub + lb >= 0))
    LA_row[p_und] = 1

    return torch.diag(UA_row), Ub, torch.diag(LA_row), Lb


if __name__ == '__main__':
    lb = torch.tensor([1, -1, -2, -2], dtype=torch.float)
    ub = torch.tensor([2, 2, 1, -1], dtype=torch.float)
    a, b, c, d = relu_relation_matrix(ub, lb)
    print(a)
    print(b)
    print(c)
    print(d)
    print('='*20)
    from collections import namedtuple
    LBound = namedtuple('LBound', ['UA', 'Ub', 'LA', 'Lb'])
    t = LBound(a, b, c, d)

    from util import bounds_prop

    w = torch.tensor([[1, -1, 0, 0], [0, 1, 1, 0], [0, -0.5, -0.5, 0]])
    bi = torch.tensor([1, 1, 0.5])
    UA, Ub, LA, Lb = bounds_prop(w, bi, a, b, c, d)
    print(UA)
    print(Ub)
    print(LA)
    print(Lb)




