import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math

def variation_ratio(predicted_probs):
    T=predicted_probs.shape[0]
    max_predictions=np.argmax(predicted_probs,axis=1)
    mode_class=stats.mode(max_predictions)[0][0]
    f_x=len((np.where(max_predictions==mode_class))[0])
    var_ratio=1-(f_x/T)
    return var_ratio

def predictive_entropy(predicted_probs):
    mean_probs=np.mean(predicted_probs,axis=0)
    entropy=get_entropy(mean_probs)
    predictive_entropy=-1 * np.sum(entropy)
    return predictive_entropy


def mutual_information(predicted_probs) :
    T = predicted_probs.shape[0]
    entropy=[get_entropy(predicted_probs[i,:]) for i in range(T)]
    exp_value=np.sum(entropy,axis=(0,1))
    mutual_information=predictive_entropy(predicted_probs)+ (exp_value/T)
    return mutual_information

def get_entropy(predicted_probs):
    entropy=np.log(predicted_probs) * predicted_probs
    nan_indexes=np.where(np.isnan(entropy))[0]
    np.put(entropy, nan_indexes, 0)
    return entropy
def prediction_variance(predicted_probs):
    return np.var(predicted_probs, axis=0)



def plot_uncertainty(uncertainty,predict_probs,adversarial_type='fgsm',epsilon=0.3):

    max_predictions=predict_probs
    variation_ratios=uncertainty['varation_ratio']
    mutual_information=uncertainty['mutual_information']
    predictive_entropy=uncertainty['predictive_entropy']
    ###################################
    # PLOT Variation Ratio Uncertainty#
    ###################################
    xy = np.vstack([max_predictions, variation_ratios])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x = max_predictions[idx]
    y = variation_ratios[idx]
    z = z[idx]
    
    fig, ax =  plt.subplots()
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    plt.colorbar(sc)
    plt.title('variation ratio uncertainty for MNIST adversarial, adversarial_type= %s,epsilon=%f'%((adversarial_type,epsilon)))
    plt.xlabel('predicted probability')
    plt.ylabel('variation ratio')
    fig.savefig('Results/VariationRatio' + '_' + adversarial_type + '_' + str(epsilon) + '.png', format='png')
    #plt.show()


    ######################################
    # PLOT MUTUAL INFORMATION Uncertainty#
    ######################################
    xy = np.vstack([max_predictions, mutual_information])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x = max_predictions[idx]
    y = mutual_information[idx]
    z = z[idx]
    
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
    plt.colorbar(sc)
    plt.title('mutual information for MNIST adversarial, adversarial_type= %s,epsilon=%f'%((adversarial_type,epsilon)))
    plt.xlabel('predicted probability')
    plt.ylabel('mutual information')
    fig.savefig('Results/MutualInformation' + '_' + adversarial_type + '_' + str(epsilon) + '.png', format='png')
    #plt.show()



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
    fig.savefig('Results/PredictedEntropy' + '_' + adversarial_type + '_' + str(epsilon) + '.png', format='png')
    #plt.show()


