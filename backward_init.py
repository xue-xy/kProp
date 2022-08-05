import torch
import numpy as np
import activation_bounds as actb
import util


def back_prop(weight, bias, relations, verbose=False):
    '''
    backward DeepPoly, calculate the relation between this layer and the first layer.
    :param weight:
    :param bias:
    :param relations: layer relations of all previous layers
    :return:
    '''
    uppers_A, uppers_b, lowers_A, lowers_b = relations
    layer_num = len(lowers_A)

    UA = weight
    Ub = bias
    LA = weight
    Lb = bias

    for i in range(layer_num - 1, -1, -1):
        upper_pos = torch.maximum(UA, torch.zeros_like(UA))
        upper_neg = torch.minimum(UA, torch.zeros_like(UA))

        UA = torch.matmul(upper_pos, uppers_A[i]) + torch.matmul(upper_neg, lowers_A[i])
        Ub = torch.matmul(upper_pos, uppers_b[i]) + torch.matmul(upper_neg, lowers_b[i]) + Ub

        lower_pos = torch.maximum(LA, torch.zeros_like(LA))
        lower_neg = torch.minimum(LA, torch.zeros_like(LA))

        LA = torch.matmul(lower_pos, lowers_A[i]) + torch.matmul(lower_neg, uppers_A[i])
        Lb = torch.matmul(lower_pos, lowers_b[i]) + torch.matmul(lower_neg, uppers_b[i]) + Lb

        if verbose:
            print(UA)
            print(Ub)
            quit()

    return UA, Ub, LA, Lb


def layer_relation(weight, bias, relations, norm, data0, radius):
    '''
    calculate the relations between the post-activation after weights and before weights
    :return: upper bounds and lower bounds list
    '''
    uppers_A, uppers_b, lowers_A, lowers_b = relations

    UA, Ub, LA, Lb = back_prop(weight, bias, relations)

    ub, lb = util.numerical_bound(UA, Ub, LA, Lb, norm, data0, radius)

    acti_UA, acti_Ub, acti_LA, acti_Lb = actb.relu_relation_simple(ub, lb)

    layer_UA = acti_UA.view(-1, 1) * weight
    layer_Ub = acti_UA * bias + acti_Ub
    layer_LA = acti_LA.view(-1, 1) * weight
    layer_Lb = acti_LA * bias + acti_Lb

    uppers_A.append(layer_UA)
    uppers_b.append(layer_Ub)
    lowers_A.append(layer_LA)
    lowers_b.append(layer_Lb)

    return uppers_A, uppers_b, lowers_A, lowers_b


def final_relation(target_A, target_b, relations):
    uppers_A, uppers_b, lowers_A, lowers_b = relations

    layer_num = len(uppers_A)

    UA = target_A
    Ub = target_b
    LA = target_A
    Lb = target_b

    for i in range(layer_num - 1, -1, -1):
        upper_pos = torch.maximum(UA, torch.zeros_like(UA))
        upper_neg = torch.minimum(UA, torch.zeros_like(UA))

        UA = torch.matmul(upper_pos, uppers_A[i]) + torch.matmul(upper_neg, lowers_A[i])
        Ub = torch.matmul(upper_pos, uppers_b[i]) + torch.matmul(upper_neg, lowers_b[i]) + Ub

        lower_pos = torch.maximum(LA, torch.zeros_like(LA))
        lower_neg = torch.minimum(LA, torch.zeros_like(LA))

        LA = torch.matmul(lower_pos, lowers_A[i]) + torch.matmul(lower_neg, uppers_A[i])
        Lb = torch.matmul(lower_pos, lowers_b[i]) + torch.matmul(lower_neg, uppers_b[i]) + Lb

    return UA, Ub, LA, Lb


def deeppoly_framework(weights, biases, testdata, testlabel, norm, radius, verbose=False):
    label = testlabel.item()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    relations = ([], [], [], [])

    for l in range(0, len(weights) - 1):
        weight = weights[l].to(device)
        bias = biases[l].to(device)

        relations = layer_relation(weight, bias, relations, norm, testdata, radius)
        # print('layer finish')

    res = np.zeros(10) - 1
    for i in range(10):
        if i != label:
            target_A = weights[-1][i] - weights[-1][label]
            target_b = biases[-1][i] - biases[-1][label]
        else:
            continue
        UA, Ub, LA, Lb = back_prop(target_A, target_b, relations)

        ub0 = util.numerical_upper_bound(UA, Ub, norm, testdata, radius)
        res[i] = ub0.item()

    if verbose:
        print('res:', res)

    if (res < 0).all():
        if verbose:
            print('verified')
        return True
    else:
        if verbose:
            print('undecidable', np.max(res))
        return False


if __name__ == '__main__':
    from data_preprocess import load_testdata
    from onnx_preprocess import get_weights

    norm = 'inf'
    radius = 0.027

    weights, biases = get_weights('model/saved_weights/mnist_6_100_paras.npy')
    data, labels = load_testdata('mnist')
    res2 = deeppoly_framework(weights, biases, data[0], labels[0], norm, radius, verbose=True)
