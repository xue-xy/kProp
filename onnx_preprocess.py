import onnx
from onnx import helper, numpy_helper
import numpy as np
import torch
import os


def mnist_linear_model(model):
    if '6_100' in model:
        path = './model/saved_model/mnist_relu_6_100.onnx'
    elif '6_200' in model:
        path = './model/saved_model/mnist_relu_6_200.onnx'
    elif '9_100' in model:
        path = './model/saved_model/mnist_relu_9_100.onnx'
    elif '9_200' in model:
        path = './model/saved_model/mnist_relu_9_200.onnx'
    else:
        raise FileNotFoundError

    onnx_model = onnx.load(path)
    weights = []
    names = []

    for t in onnx_model.graph.initializer:
        weights.append(numpy_helper.to_array(t))
        names.append(t.name)

    arranged_paras = []
    arranged_names = []

    for i in range(1, len(weights)//2 + 1):
        bid = names.index(str(2*i) + '.bias')
        wid = names.index(str(2*i) + '.weight')
        arranged_names.extend([names[bid], names[wid]])
        arranged_paras.extend([weights[bid], weights[wid]])

    # for i in range(len(arranged_names)):
    #     print(arranged_names[i], arranged_paras[i].shape)

    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t)[0, :, :, 0]
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t)[0, :, :, 0]
    prew = np.diag(np.tile(1 / c2, (1, 28 * 28)).flatten('F'))
    preb = np.tile(-1 * c0 / c2, (1, 28 * 28)).flatten('F')

    arranged_paras[0] = arranged_paras[0] + np.matmul(arranged_paras[1], preb)
    arranged_paras[1] = np.matmul(arranged_paras[1], prew)

    np.save('./model/saved_weights/' + model + '_paras.npy', arranged_paras)
    return get_weights('./model/saved_weights/' + model + '_paras.npy')


def cifar_linear_model(model):
    if '6_100' in model:
        path = './model/saved_model/cifar_relu_6_100.onnx'
    elif '9_200' in model:
        path = './model/saved_model/cifar_relu_9_200.onnx'
    else:
        raise FileNotFoundError

    onnx_model = onnx.load(path)
    weights = []
    names = []

    for t in onnx_model.graph.initializer:
        weights.append(numpy_helper.to_array(t))
        names.append(t.name)

    arranged_paras = []
    arranged_names = []

    for i in range(1, len(weights)//2 + 1):
        bid = names.index(str(2*i) + '.bias')
        wid = names.index(str(2*i) + '.weight')
        arranged_names.extend([names[bid], names[wid]])
        arranged_paras.extend([weights[bid], weights[wid]])

    # for i in range(len(arranged_names)):
    #     print(arranged_names[i], arranged_paras[i].shape)

    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t)[0, :, :, 0]
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t)[0, :, :, 0]
    prew = np.diag(np.tile(1 / c2, (1, 32 * 32)).flatten('F'))
    preb = np.tile(-1 * c0 / c2, (1, 32 * 32)).flatten('F')

    arranged_paras[0] = arranged_paras[0] + np.matmul(arranged_paras[1], preb)
    arranged_paras[1] = np.matmul(arranged_paras[1], prew)

    np.save('./model/saved_weights/' + model + '_paras.npy', arranged_paras)
    return get_weights('./model/saved_weights/' + model + '_paras.npy')


def mnist_conv_model():
    onnx_model = onnx.load('./model/saved_model/mnist_convSmallRELU__Point.onnx')

    raw_paras = []
    for e in onnx_model.graph.initializer:
        raw_paras.append(numpy_helper.to_array(e))
    raw_bias = [raw_paras[i] for i in range(0, len(raw_paras), 2)]
    raw_weights = [raw_paras[i] for i in range(1, len(raw_paras), 2)]

    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t)[0, :, :, 0]
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t)[0, :, :, 0]
    prew = np.diag(np.tile(1 / c2, (1, 28*28)).flatten())
    preb = np.tile(-1*c0 / c2, (1, 28*28)).flatten()

    paras = []

    w1 = np.zeros((16*13*13, 1*28*28))
    for och in range(16):
        for ich in range(1):
            pattern = np.zeros(28*3+4)
            for j in range(4):
                pattern[j*28:j*28+4] = raw_weights[0][och, ich, j]
            for row in range(13):
                for col in range(13):
                    w1[och*13*13+row*13+col, ich*28*28+2*row*28+col*2: ich*28*28+2*row*28+col*2+3*28+4] = pattern
    b1 = np.zeros(16*13*13)
    for i in range(16):
        b1[i*13*13:(i+1)*13*13] = raw_bias[0][i]

    b1 = b1 + np.matmul(w1, preb)
    w1 = np.matmul(w1, prew)

    paras.append(b1)
    paras.append(w1)

    w2 = np.zeros((800, 16*13*13))
    for och in range(32):
        for ich in range(16):
            pattern = np.zeros(13*3+4)
            for j in range(4):
                pattern[j*13:j*13+4] = raw_weights[1][och, ich, j]
            for row in range(5):
                for col in range(5):
                    w2[och*25+row*5+col, ich*13*13+2*row*13+2*col:ich*13*13+2*row*13+2*col+13*3+4] = pattern
    b2 = np.zeros(800)
    for i in range(32):
        b2[i*25:i*25+25] = raw_bias[1][i]
    paras.append(b2)
    paras.append(w2)
    paras.extend([raw_bias[2], raw_weights[2], raw_bias[3], raw_weights[3]])

    np.save('./model/saved_weights/mnist_conv_paras.npy', paras)
    return get_weights('./model/saved_weights/mnist_conv_paras.npy')


def cifar_conv_model():
    onnx_model = onnx.load('./model/saved_model/cifar_convSmallRELU__PGDK.onnx')

    raw_paras = []
    for e in onnx_model.graph.initializer:
        raw_paras.append(numpy_helper.to_array(e))
    raw_bias = [raw_paras[i] for i in range(0, len(raw_paras), 2)]
    raw_weights = [raw_paras[i] for i in range(1, len(raw_paras), 2)]

    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t)[0, :, :, 0]
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t)[0, :, :, 0]
    prew = np.diag(np.tile(1 / c2, (1, 32*32)).flatten())
    preb = np.tile(-1*c0 / c2, (1, 32*32)).flatten()

    paras = []

    w1 = np.zeros((16*15*15, 3*32*32))
    for och in range(16):
        for ich in range(3):
            pattern = np.zeros(32*3+4)
            for j in range(4):
                pattern[j*32:j*32+4] = raw_weights[0][och, ich, j]
            for row in range(15):
                for col in range(15):
                    w1[och*15*15+row*15+col, ich*32*32+2*row*32+col*2: ich*32*32+2*row*32+col*2+3*32+4] = pattern
    b1 = np.zeros(16*15*15)
    for i in range(16):
        b1[i*15*15:(i+1)*15*15] = raw_bias[0][i]

    b1 = b1 + np.matmul(w1, preb)
    w1 = np.matmul(w1, prew)

    paras.append(b1)
    paras.append(w1)

    w2 = np.zeros((1152, 16*15*15))
    for och in range(32):
        for ich in range(16):
            pattern = np.zeros(15*3+4)
            for j in range(4):
                pattern[j*15:j*15+4] = raw_weights[1][och, ich, j]
            for row in range(6):
                for col in range(6):
                    w2[och*36+row*6+col, ich*15*15+2*row*15+2*col:ich*15*15+2*row*15+2*col+15*3+4] = pattern
    b2 = np.zeros(1152)
    for i in range(32):
        b2[i*36:i*36+36] = raw_bias[1][i]
    paras.append(b2)
    paras.append(w2)
    paras.extend([raw_bias[2], raw_weights[2], raw_bias[3], raw_weights[3]])

    np.save('./model/saved_weights/cifar_conv_paras.npy', paras)
    return get_weights('./model/saved_weights/cifar_conv_paras.npy')


def get_weights(path):
    paras = np.load(path, allow_pickle=True)
    weights = [torch.tensor(paras[i]) for i in range(1, len(paras), 2)]
    biases = [torch.tensor(paras[i]) for i in range(0, len(paras), 2)]

    return weights, biases


def linear_forward(weights, biases, x):
    # x = x.reshape(-1, 1)
    for i in range(len(weights)):
        x = np.matmul(weights[i], x) + biases[i]
        x = np.maximum(x, np.zeros_like(x))

    return x


def relu(x):
    return np.maximum(x, np.zeros_like(x))


def linear_check(data, label, weights, biases):
    pred = linear_forward(weights, biases, data.numpy())
    if np.argmax(pred) == label.item():
        return True
    else:
        return False


def para_change():
    p = np.load('./model/saved_weights/mnist_convsmall_paras_old.npy', allow_pickle=True)
    onnx_model = onnx.load('./model/saved_model/mnist_convSmallRELU__Point.onnx')

    w = [p[i] for i in range(1, len(p), 2)]
    b = [p[i] for i in range(0, len(p), 2)]
    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t).item()
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t).item()

    b[0] = b[0] - np.matmul(w[0], (np.zeros(784) + c0)) / c2
    w[0] = w[0]/c2

    paras = []
    for i in range(len(w)):
        paras.append(b[i])
        paras.append(w[i])

    # np.save('./model/saved_weights/mnist_conv_paras.npy', paras)


def get_paras(model):
    paras_path = './model/saved_weights/' + model + '_paras.npy'
    if os.path.exists(paras_path):
        return get_weights(paras_path)
    else:
        print('-+-')
        if 'cifar' in model:
            if 'conv' in model:
                return cifar_conv_model()
            else:
                return cifar_linear_model(model)
        elif 'mnist' in model:
            if 'conv' in model:
                return mnist_conv_model()
            else:
                return mnist_linear_model(model)


if __name__ == '__main__':
    from data_preprocess import load_testdata
    images, labels = load_testdata('mnist')

    w, b = mnist_conv_model()
    c = 0
    for i in range(1000):
        if linear_check(images[i], labels[i], w, b):
            c += 1
    print(c)

