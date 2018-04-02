import numpy as np
from scipy import stats
import math

def variation_ratio(predicted_probs):
    T=predicted_probs.shape[0]
    max_predictions=np.argmax(predicted_probs,axis=1)
    mode_class=stats.mode(max_predictions)[0][0]
    f_x=len((np.where(max_predictions==mode_class))[0])
    var_ratio=1-(f_x/T)
    if np.isnan(var_ratio):
        print('hi')
    if math.isnan(var_ratio):
        print('hi')
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

