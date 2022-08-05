import torch
from scipy.optimize import linprog
import numpy as np


def numerical_bound(U_A, U_b, L_A, L_b, norm, x0, perturbation, trim=True):
    """
    If trim is true, cut the regions outside [0,1]
    :param U_A:
    :param U_b:
    :param L_A:
    :param L_b:
    :param norm:
    :param x0:
    :param perturbation:
    :param trim:
    :return:
    """
    if trim:
        # x0 = x0.double()
        variables_low = torch.clip(x0 - perturbation, 0.0, 1.0)
        variables_up = torch.clip(x0 + perturbation, 0.0, 1.0)
        if norm == 'inf':
            ub = torch.matmul(torch.maximum(U_A, torch.zeros_like(U_A)), variables_up) + \
                 torch.matmul(torch.minimum(U_A, torch.zeros_like(U_A)), variables_low) + U_b
            lb = torch.matmul(torch.maximum(L_A, torch.zeros_like(L_A)), variables_low) + \
                 torch.matmul(torch.minimum(L_A, torch.zeros_like(L_A)), variables_up) + L_b
        elif norm == '1':
            up_range = torch.minimum(torch.ones_like(x0), x0 + perturbation) - x0
            low_range = x0 - torch.maximum(torch.zeros_like(x0), x0 - perturbation)
            # upper_bound_range = (U_A >= 0) * up_range + (U_A < 0) * low_range
            # lower_bound_range = (U_A >= 0) * low_range + (U_A < 0) * up_range
            _, indices = torch.sort(torch.abs(U_A), descending=True)
            ub = torch.zeros(U_A.shape[0])
            lb = torch.zeros(U_A.shape[0])
            for i in range(U_A.shape[0]):
                # ub[i] = in_region_up_bound_1(U_A[i], upper_bound_range[i], indices[i], perturbation) + U_b[i]
                # lb[i] = -1 * in_region_up_bound_1(U_A[i], lower_bound_range[i], indices[i], perturbation) + L_b[i]
                ub[i] = clip_up_bound1_loop(U_A[i], up_range, low_range, indices[i], perturbation) + U_b[i]
                lb[i] = clip_low_bound1_loop(U_A[i], up_range, low_range, indices[i], perturbation) + L_b[i]
            ub = ub + torch.matmul(U_A, x0)
            lb = lb + torch.matmul(L_A, x0)
        elif norm == '2':
            up_range = (1 - x0) * (U_A > 0) + (- x0) * (U_A < 0)
            low_range = (- x0) * (U_A > 0) + (1 - x0) * (U_A < 0)

            _, up_indices = torch.sort(up_range / U_A, descending=False)
            _, low_indices = torch.sort(low_range / U_A, descending=True)

            ub = torch.zeros(U_A.shape[0])
            lb = torch.zeros(U_A.shape[0])
            for i in range(U_A.shape[0]):
                ub[i] = clip_up_bound2_loop(U_A[i], up_range[i], up_indices[i], perturbation)
                lb[i] = clip_low_bound2_loop(L_A[i], low_range[i], low_indices[i], perturbation)
            ub = ub + torch.matmul(U_A, x0) + U_b
            lb = lb + torch.matmul(L_A, x0) + L_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)
    else:
        if U_A.dim() == 2:
            d = 1
        elif U_A.dim() == 1:
            d = 0

        if norm == '1':
            ub = torch.linalg.norm(U_A, ord=float('inf'), dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
            lb = -1 * torch.linalg.norm(L_A, ord=float('inf'), dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        elif norm == '2':
            ub = torch.linalg.norm(U_A, ord=2, dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
            lb = -1 * torch.linalg.norm(L_A, ord=2, dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        elif norm == 'inf':
            ub = torch.linalg.norm(U_A, ord=1, dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
            lb = -1 * torch.linalg.norm(L_A, ord=1, dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)

    return ub, lb


def numerical_upper_bound(U_A, U_b, norm, x0, perturbation, trim=True):
    if trim:
        # x0 = x0.double()
        if norm == 'inf':
            variables_low = torch.clip(x0 - perturbation, 0.0, 1.0)
            variables_up = torch.clip(x0 + perturbation, 0.0, 1.0)
            ub = torch.matmul(torch.maximum(U_A, torch.zeros_like(U_A)), variables_up) + \
                 torch.matmul(torch.minimum(U_A, torch.zeros_like(U_A)), variables_low) + U_b
        elif norm == '1':
            up_range = torch.minimum(torch.ones_like(x0), x0 + perturbation) - x0
            low_range = x0 - torch.maximum(torch.zeros_like(x0), x0 - perturbation)
            # useful_range = (U_A >= 0) * up_range + (U_A < 0) * low_range
            _, indices = torch.sort(torch.abs(U_A), descending=True)

            if U_A.dim() == 2:
                ub = torch.zeros(U_A.shape[0])
                for i in range(U_A.shape[0]):
                    ub[i] = clip_up_bound1_loop(U_A[i], up_range, low_range, indices[i], perturbation) + U_b[i]
                ub = ub + torch.matmul(U_A, x0)
            elif U_A.dim() == 1:
                ub = clip_up_bound1_loop(U_A, up_range, low_range, indices, perturbation) + torch.matmul(U_A, x0) + U_b
        elif norm == '2':
            feasible = (1 - x0) * (U_A > 0) + (- x0) * (U_A < 0)
            _, indices = torch.sort(feasible / U_A, descending=False)

            if U_A.dim() == 2:
                ub = torch.zeros(U_A.shape[0])
                for i in range(U_A.shape[0]):
                    ub[i] = clip_up_bound2_loop(U_A[i], feasible[i], indices[i], perturbation)
                ub = ub + torch.matmul(U_A, x0) + U_b
            elif U_A.dim() == 1:
                ub = clip_up_bound2_loop(U_A, feasible, indices, perturbation) + torch.matmul(U_A, x0) + U_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)
    else:
        if U_A.dim() == 2:
            d = 1
        elif U_A.dim() == 1:
            d = 0

        if norm == '1':
            ub = torch.linalg.norm(U_A, ord=float('inf'), dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
        elif norm == '2':
            ub = torch.linalg.norm(U_A, ord=2, dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
        elif norm == 'inf':
            ub = torch.linalg.norm(U_A, ord=1, dim=d) * perturbation + torch.matmul(U_A, x0) + U_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)

    return ub


def numerical_lower_bound(L_A, L_b, norm, x0, perturbation, trim=True):
    if trim:
        variables_low = torch.clip(x0 - perturbation, 0.0, 1.0)
        variables_up = torch.clip(x0 + perturbation, 0.0, 1.0)
        if norm == 'inf':
            lb = torch.matmul(torch.maximum(L_A, torch.zeros_like(L_A)), variables_low) + \
                 torch.matmul(torch.minimum(L_A, torch.zeros_like(L_A)), variables_up) + L_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)
    else:
        if L_A.dim() == 2:
            d = 1
        elif L_A.dim() == 1:
            d = 0

        if norm == '1':
            lb = -1 * torch.linalg.norm(L_A, ord=float('inf'), dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        elif norm == '2':
            lb = -1 * torch.linalg.norm(L_A, ord=2, dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        elif norm == 'inf':
            lb = -1 * torch.linalg.norm(L_A, ord=1, dim=d) * perturbation + torch.matmul(L_A, x0) + L_b
        else:
            raise NotImplementedError('norm %s is not supported' % norm)

    return lb


def bounds_prop(coef_A, coef_b, upper_A, upper_b, lower_A, lower_b):
    assert coef_A.dim() == 2, 'Wrong dimension. 2 needed, but {} given.'.format(coef_A.dim())

    pos = torch.maximum(coef_A, torch.zeros_like(coef_A))
    neg = torch.minimum(coef_A, torch.zeros_like(coef_A))

    UA = torch.mm(pos, upper_A) + torch.mm(neg, lower_A)
    Ub = torch.matmul(pos, upper_b) + torch.matmul(neg, lower_b) + coef_b
    LA = torch.mm(pos, lower_A) + torch.mm(neg, upper_A)
    Lb = torch.matmul(pos, lower_b) + torch.matmul(neg, upper_b) + coef_b

    return UA, Ub, LA, Lb


def clip_up_bound1_loop(coefficient, u_range, l_range, idx, eps):
    remain = eps
    res = 0
    for i in range(idx.shape[0]):
        if coefficient[idx[i]] >= 0:
            if u_range[idx[i]] < remain:
                res += coefficient[idx[i]] * u_range[idx[i]]
                remain -= u_range[idx[i]]
            else:
                res += coefficient[idx[i]] * remain
                break
        else:
            if u_range[idx[i]] < remain:
                res += coefficient[idx[i]] * (-1 * l_range[idx[i]])
                remain -= l_range[idx[i]]
            else:
                res += coefficient[idx[i]] * (-1 * remain)
                break
    return res


def clip_low_bound1_loop(coefficient, u_range, l_range, idx, eps):
    remain = eps
    res = 0
    for i in range(idx.shape[0]):
        if coefficient[idx[i]] >= 0:
            if u_range[idx[i]] < remain:
                res += coefficient[idx[i]] * (-1 * l_range[idx[i]])
                remain -= l_range[idx[i]]
            else:
                res += coefficient[idx[i]] * (-1 * remain)
                break
        else:
            if u_range[idx[i]] < remain:
                res += coefficient[idx[i]] * u_range[idx[i]]
                remain -= u_range[idx[i]]
            else:
                res += coefficient[idx[i]] * remain
                break
    return res


def clip_up_bound2_loop(coefficients, v_range, idx, eps):
    remain = eps * eps
    res = 0
    coefficients = coefficients[idx]
    v_range = v_range[idx]
    for i in range(idx.shape[0]):
        if v_range[i] == 0:
            continue
        if torch.sum(coefficients[i:] ** 2) * (v_range[i] ** 2) / (coefficients[i] ** 2) >= remain:
            res += torch.sqrt(torch.sum(coefficients[i:] ** 2) * remain)
            break
        else:
            res += coefficients[i] * v_range[i]
            remain = remain - (v_range[i] ** 2)
    return res


def clip_low_bound2_loop(coefficients, v_range, idx, eps):
    """
    v_range is different from clip_up_bound2_loop and take the opposite direction.
    :param coefficients:
    :param v_range:
    :param idx:
    :param eps:
    :return:
    """
    remain = eps * eps
    res = 0
    coefficients = coefficients[idx]
    v_range = v_range[idx]
    for i in range(idx.shape[0]):
        if v_range[i] == 0:
            continue
        if torch.sum(coefficients[i:] ** 2) * (v_range[i] ** 2) / (coefficients[i] ** 2) >= remain:
            res = res - torch.sqrt(torch.sum(coefficients[i:] ** 2) * remain)
            break
        else:
            res += coefficients[i] * v_range[i]
            remain = remain - (v_range[i] ** 2)
    return res


if __name__ == '__main__':
    torch.random.manual_seed(1)
    w = torch.rand((4, 5)) - 0.5
    d = torch.tensor([0, 0.3, 0.5, 0.8, 1])
    ep = 1.5

    up_range = (1 - d) * (w > 0) + (- d) * (w < 0)
    low_range = (- d) * (w > 0) + (1 - d) * (w < 0)

    ub, lb = numerical_bound(w, torch.zeros(4), w, torch.zeros(4), '2', d, 1, trim=True)
    print(torch.matmul(w, d))
    print(ub)
    ub, lb = numerical_bound(w, torch.zeros(4), w, torch.zeros(4), '2', d, 1, trim=False)
    print(ub)
    # ub, lb = numerical_bound(w, torch.zeros(4), w,torch.zeros(4), '1', d, 2, trim=False)
    # print(ub)
    # print(lb)
    # ub, lb = numerical_bound(w, torch.zeros(4), w, torch.zeros(4), '1', d, 2)
    # print(ub)
    # print(lb)
    # print(torch.sum(useful_range[0][indices[0][:2]]))
