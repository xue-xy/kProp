import numpy as np
from tqdm import tqdm
from time import time
import argparse

from data_preprocess import load_testdata
from backward_kpoly import kprop_framework
from onnx_preprocess import get_paras, linear_check


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mnist_6_100', 'mnist_6_200', 'mnist_9_100', 'mnist_9_200', 'mnist_conv',
                                            'cifar_6_100', 'cifar_9_200', 'cifar_conv'],
                        help='saved model')
    parser.add_argument('-n', '--norm', type=str, help='norm')
    parser.add_argument('-r', '--radius', type=float, help='neighborhood radius')
    parser.add_argument('-k', type=int, default=3, help='number of neurons in a group')

    args = parser.parse_args()
    model = args.model
    norm = args.norm
    radius = args.radius
    group_num = args.k

    if 'mnist' in model:
        data, labels = load_testdata('mnist')
    elif 'cifar' in model:
        if 'conv' in model:
            data, labels = load_testdata('cifar_c')
        else:
            data, labels = load_testdata('cifar')
    else:
        print('please load your dataset')
    weights, biases = get_paras(model)

    weights[0] = weights[0].float()
    biases[0] = biases[0].float()
    weights[1] = weights[1].float()
    biases[1] = biases[1].float()

    kprop_res = np.zeros(data.shape[0])

    for i in tqdm(range(1)):
        if not linear_check(data[i], labels[i], weights, biases):
            continue

        kprop_r = kprop_framework(weights, biases, data[i], labels[i], norm, radius, group_num)
        if kprop_r:
            kprop_res[i] = 1

    # print('kprop result:', kprop_res)
    print('kprop verified:', np.sum(kprop_res))