import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


dir='../cleverhans_core/examples/'
def get_adversarial_examples(dataset='MNIST',adversarial_type='fgsm',epsilon=0.3,number=1000):
    if dataset=='MNIST':
        return get_MNIST_adversarial(epsilon,number,adversarial_type)



def get_MNIST_adversarial(epsilon,number,adversarial_type='fgsm'):
    if adversarial_type=='fgsm':
        filename=('%sfgsm_mnist_examples_x_1000_%s.npy'%(dir,str(epsilon)))
        adversarial_MNIST = np.load(filename)
        true_labels=get_MNIST_true_labels(number)
        return  adversarial_MNIST,true_labels


def get_MNIST_true_labels(number):
    train_dataset = dsets.MNIST(root='./input/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    return train_dataset.train_labels[0:number].numpy()

