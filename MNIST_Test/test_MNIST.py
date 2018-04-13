import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from  MNIST_Net import Net
from MNIST_Test import Uncertainty,generate_adversarials

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#hyper parameters
T=100


# Pass in the 28x28x1 tensor for the image
def show_Image(image_data):
    plt.imshow(image_data[:,:,0], cmap='gray')
    plt.show()

def remove_nan_values(adv_predictions):
    new_adv_predictions=np.copy(adv_predictions)
    for i in range(adv_predictions.shape[0]):
        if np.sum(np.isnan(adv_predictions[i,:]))>0:
            new_adv_predictions=np.delete(adv_predictions, i, 0)
    return new_adv_predictions
def get_adversarial_predictions(adversarial_set,cnn):
    adversarial_MNIST = adversarial_set
    n = adversarial_MNIST.shape[0]
    max_predictions = np.zeros(n)
    predicted_labels=np.zeros(n)
    adv_variation_ratios = np.zeros(n)
    adv_predictive_entropy = np.zeros(n)
    adv_mutual_information = np.zeros(n)
    softmax = nn.Softmax()

    for index, adv in enumerate(adversarial_MNIST):
        # show_Image(np.squeeze(adv))
        adv_squeezed = torch.squeeze(torch.from_numpy(adv).float())
        adv_tensor = torch.FloatTensor(T, 1, adv_squeezed.size()[0], adv_squeezed.size()[1])
        for i in range(T):
            adv_tensor.add(adv_squeezed.unsqueeze(0))
        adversarial_example = Variable(adv_tensor)
        adv_outputs = cnn(adversarial_example)
        adv_predictions = softmax(adv_outputs).data.numpy()
        new_adv_predictions=remove_nan_values(adv_predictions)
        mean = np.mean(new_adv_predictions, axis=0)
        max_predictions[index] = np.max(mean)
        predicted_labels[index]=np.argmax(mean)
        adv_variation_ratios[index] = Uncertainty.variation_ratio(new_adv_predictions)
        adv_predictive_entropy[index] = Uncertainty.predictive_entropy(new_adv_predictions)
        adv_mutual_information[index] = Uncertainty.mutual_information(new_adv_predictions)

    uncertainty={}
    uncertainty['varation_ratio']=adv_variation_ratios
    uncertainty['predictive_entropy']=adv_predictive_entropy
    uncertainty['mutual_information']=adv_mutual_information

    return uncertainty,predicted_labels,max_predictions

def plot_uncertainty(uncertainty,predict_probs,adversarial_type='fgsm',epsilon=0.3):

    max_predictions=predict_probs
    variation_ratios=uncertainty['varation_ratio']
    mutual_information=uncertainty['mutual_information']
    predictive_entropy=uncertainty['predictive_entropy']
    ###################################
    # PLOT Variation Ratio Uncertainty#
    ###################################
    # xy = np.vstack([max_predictions, variation_ratios])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x = max_predictions[idx]
    # y = variation_ratios[idx]
    # z = z[idx]
    #
    # fig, ax = plt.subplots()
    # sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    # plt.colorbar(sc)
    # plt.title('variation ratio uncertainty for MNIST adversarial, adversarial_type= %s,epsilon=%f'%((adversarial_type,epsilon)))
    # plt.xlabel('predicted probability')
    # plt.ylabel('variation ratio')
    # plt.show()


    ######################################
    # PLOT MUTUAL INFORMATION Uncertainty#
    ######################################
    # xy = np.vstack([max_predictions, mutual_information])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x = max_predictions[idx]
    # y = mutual_information[idx]
    # z = z[idx]
    #
    # fig, ax = plt.subplots()
    # sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    # plt.colorbar(sc)
    # plt.title('mutual information for MNIST adversarial, adversarial_type= %s,epsilon=%f'%((adversarial_type,epsilon)))
    # plt.xlabel('predicted probability')
    # plt.ylabel('mutual information')
    # plt.show()



    ######################################
    # PLOT PREDICTIVE ENTROPY Uncertainty#
    ######################################
    xy = np.vstack([max_predictions, predictive_entropy])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x = max_predictions[idx]
    y = predictive_entropy[idx]
    z = z[idx]

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    plt.colorbar(sc)
    plt.title('predictive entropy for MNIST adversarial,adversarial_type= %s,epsilon=%f'%((adversarial_type,epsilon)))
    plt.xlabel('predicted probability')
    plt.ylabel('predictive entropy')
    plt.show()

def get_testset_predictions():
    softmax=nn.Softmax()
    test_dataset = dsets.MNIST(root='./input/',
                               train=False,
                               transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    t = test_dataset.test_data.shape[0]
    test_max_predictions = np.zeros(t)
    test_variation_ratios = np.zeros(t)
    test_predictive_entropy = np.zeros(t)
    test_mutual_information = np.zeros(t)

    for index, (images, labels) in enumerate(test_loader):
        images = torch.squeeze(images)
        img_tensor = torch.FloatTensor(T, 1, images.size()[0], images.size()[0])
        for i in range(T):
            img_tensor.add(images.unsqueeze(0))
        img_tensor = Variable(img_tensor)
        outputs = cnn(img_tensor)
        predictions = softmax(outputs).data.numpy()
        test_max_predictions[index] = np.max(np.mean(predictions, axis=0))
        test_variation_ratios[index] = Uncertainty.variation_ratio(predictions)
        test_predictive_entropy[index] = Uncertainty.predictive_entropy(predictions)
        test_mutual_information[index] = Uncertainty.mutual_information(predictions)

    # Sort the points by density, so that the densest points are plotted last
    xy = np.vstack([test_max_predictions, test_mutual_information])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x = test_max_predictions[idx]
    y = test_mutual_information[idx]
    z = z[idx]

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    plt.colorbar(sc)
    plt.title('probability of predicted class-vs-mutual information for MNIST test set')
    plt.xlabel('predicted probability')
    plt.ylabel('mutual information')
    plt.show()


if __name__ == '__main__':
    
    # load the Trained model
    cnn = Net()
    print('start loading model')
    cnn.load_state_dict(torch.load('cnn.pkl'))
    print('end loading model')

    #######################################
    ##########  TEST on FGSM ##############
    #######################################
    epsilon=0.3

    adversarial_set,true_labels=generate_adversarials.get_adversarial_examples(dataset='MNIST',adversarial_type='fgsm',epsilon=0.3,number=1000)

    uncertainty,predicted_labels,predict_probs=get_adversarial_predictions(adversarial_set,cnn)

    # plot_uncertainty(uncertainty,predict_probs,adversarial_type='fgsm',epsilon=0.3)

    error_rate=1-(np.sum(np.equal(true_labels,predicted_labels))/1000)

    print('model error on adversarial examples: %f'%error_rate)

