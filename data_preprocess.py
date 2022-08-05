import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


def load_testdata(datatype):
    if datatype == 'mnist':
        data = np.loadtxt('./data/mnist_1000.csv', delimiter=',')
        labels = torch.tensor(data[:, 0], dtype=torch.int)
        images = torch.tensor(data[:, 1:], dtype=torch.float) / 255
    elif datatype == 'cifar':
        data = np.loadtxt('./data/cifar_1000.csv', delimiter=',')
        labels = torch.tensor(data[:, 0], dtype=torch.int)
        images = torch.tensor(data[:, 1:], dtype=torch.float) / 255
    elif datatype == 'cifar_c':
        data = np.loadtxt('./data/cifar_1000_c.csv', delimiter=',')
        labels = torch.tensor(data[:, 0], dtype=torch.int)
        images = torch.tensor(data[:, 1:], dtype=torch.float) / 255
    else:
        raise FileNotFoundError('{} not supported'.format(datatype))

    return images, labels


def mnist_test_dataset():
    test_dataset = datasets.MNIST(root='./data', train=False, transform=ToTensor())
    test_data = torch.flatten(test_dataset.data, 1).numpy()
    test_label = test_dataset.targets.numpy()
    data = np.concatenate([test_label.reshape(-1, 1), test_data], axis=1)

    # np.random.seed()
    # idx = np.random.choice(data.shape[0], 100)
    # selected = data[idx]
    np.savetxt('./data/mnist_1000.csv', data[:1000], delimiter=',')


def cifar_test_dataset():
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=ToTensor())
    raw_data = test_dataset.data
    test_data = np.zeros((raw_data.shape[0], 3072))
    for i in range(3):
        test_data[:, i*1024:(i+1)*1024] = torch.flatten(torch.tensor(raw_data[:, :, :, i]), start_dim=1).numpy()
    # test_d = torch.flatten(torch.tensor(test_dataset.data), 1).numpy()
    test_label = np.array(test_dataset.targets)
    data = np.concatenate([test_label.reshape(-1, 1), test_data], axis=1)

    np.savetxt('./data/cifar_1000_c.csv', data[:1000], delimiter=',')


def get_paras(net):
    paras = net.state_dict()
    keys = list(paras.keys())

    weights = [paras[keys[x]] for x in range(0, len(keys), 2)]
    biases = [paras[keys[x]] for x in range(1, len(keys), 2)]

    return weights, biases


if __name__ == '__main__':
    # data, label = load_testdata('cifar')
    # print(data.shape)
    # print(torch.max(data))
    # print(label[0])
    # mnist_test_dataset()
    cifar_test_dataset()