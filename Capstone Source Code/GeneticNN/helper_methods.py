#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   helper_methods.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np
from network import Network
import random
def downsample(image):
    return image[::2, ::2, :]
def remove_color(image):
    """ upon inspection the third dimension is the RGB value"""
    return image[:, :, 0]
def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image
def preprocess_observation(input_obs, pre_processed_obs, input_dimension):
    """Remove the score from the input,  """
    processed_obs = input_obs[35:195]
    processed_obs = downsample(processed_obs)
    processed_obs = remove_color(processed_obs)
    processed_obs = remove_background(processed_obs)
    processed_obs[processed_obs != 0] = 1 # anything that is not zero set it to 1(normalizaton step)

    processed_obs = processed_obs.astype(np.float).ravel()

    if pre_processed_obs is not None:
        input_obs = processed_obs - pre_processed_obs
    else:
        input_obs = np.zeros((input_dimension,1))
    prev_processed_obser = processed_obs
    return input_obs, prev_processed_obser


def _makeDerivativeMatrix(self, index, a):
    """
        Computes the corresponding jaccbian matrix for the derivative matrix
        F^1(n-1)

        Precondition: feed forward must've been run prior to this function call
        :param index:
        :return: None
        """
    num_neurons = a
    jaccob_matrix = np.zeros(shape=(num_neurons, num_neurons))  # ie S=3, shape 3X3
    # dx_func = self.__getDerivative(self.layers[index]['trans_func'])
    dx_func = self._getTransFunc(self.layers[index]['trans_func']).derivative
    for i in range(num_neurons):
        # diagonal matrix
        a_val = self.layers[index]['a_output'][i]
        jaccob_matrix[i][i] = dx_func(a_val)
    return jaccob_matrix


def update_weights(weights, exp_g_sq, g_dict, decay_r, learn_r):
    # done
    eps = 1e-5
    for layer in weights.keys():
        g = g_dict[layer]
        exp_g_sq[layer] = decay_r * exp_g_sq[layer] + (1 - decay_r) * g ** 2
        print("Weights :", weights[layer].shape)
        print("g : ", g.shape)
        print("exp_g_sqrt: ", exp_g_sq[layer].shape)
        weights[layer] += (learn_r * g) / (np.sqrt(exp_g_sq[layer] + eps))

        g_dict[layer] = np.zeros_like(weights[layer])


def relu(vector):
    # done
    vector[vector < 0] = 0
    return vector


def compute_gradient(gradient_log_p, hidden_layer_val, obs_val, weights):
    delta_L = gradient_log_p

    dxC_dw2 = np.dot(hidden_layer_val.T, delta_L).ravel()
    print("delta_L : ", delta_L.shape)
    print("dxC_dw2: ", dxC_dw2.shape)
    print("Layer 2 Weights : ", weights['2'].shape)

    delta_l2 = np.outer(delta_L, weights['2'])  # last layer, output layer
    print("delta_l2: ", delta_l2.shape)
    delta_l2 = relu(delta_l2)
    dxC_dw1 = np.dot(delta_l2.T, obs_val)
    print("dxC_dw1: ", dxC_dw1.shape)
    return {
        '1': dxC_dw1,
        '2': dxC_dw2
    }


def discount_reward(reward, gamma):
    """Discount ensures that actions that were taken in the beginning weight less of an
        importance to the result than actions that were taken 2 steps prior to this result"""
    discount_r = np.zeros_like(reward)
    r_total = 0
    for _ in reversed(range(0, reward.size)):
        if reward[_] != 0:
            r_total = 0
        r_total = r_total * gamma + reward[_]
        discount_r[_] = r_total
    return discount_r


def discount_with_rewards(gradient_log, ep_rewards, gamma):
    discount_ep_reward = discount_reward(ep_rewards, gamma)

    discount_ep_reward -= np.mean(discount_ep_reward)
    discount_ep_reward /= np.std(discount_ep_reward)
    return gradient_log * discount_ep_reward

def mutate(network):
    #given a network randomly choose a parameter and randomize that parameter
    param_to_change = random.choice(list(network.getHyperParam().keys()))
    if param_to_change == 'learning_rate' or param_to_change == 'decay_rate':
        network.getHyperParam()[param_to_change] = random.random()
    elif param_to_change == 'weights' or param_to_change == 'num_hidden_neuron':
#if we run into hidden layer change OR weight change we need to reinitatlize the weights , because the dimensions will need to be updated
        network.getHyperParam()['num_hidden_neuron'] = random.randint(120, 700)
        network.reinitWeights() #after changing the number of hidden neurons we need to reintalze the shape of our weights

def generateChildren(parents,num_children):
    spawn = []
    for i in range(num_children):
        _param = {}
        for param in parents[0].getHyperParam().keys():
            #if param == 'weights':
            #    _param['weights'] = {}
            #    print("comparing weights %s" % param)
            #    for w in parents[0].getHyperParam()[param].keys():
            #        _param[param][w] = random.choice(parents).getHyperParam()[param][w]
            #else:
            #    _param[param] = random.choice(parents).getHyperParam()[param]
            _param[param] = random.choice(parents).getHyperParam()[param]
        _net = Network(_param)
        spawn.append(_net)
    return spawn