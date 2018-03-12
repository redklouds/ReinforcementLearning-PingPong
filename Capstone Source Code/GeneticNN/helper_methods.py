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


def preprocess_observation(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195]  # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    #print(processed_observation.shape)
    processed_observation = processed_observation.astype(np.float).ravel()
    processed_observation = processed_observation.reshape((1,input_dimensions))
    #print(processed_observation.shape)
    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros( (1, input_dimensions) )
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    #print("input observation size ", input_observation.shape)
    #print("RETURNING %s" % input_observation.shape)
    return input_observation, prev_processed_observations

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

        #print("UPDATE Layer: ", layer)
        # print("UPDATE weights ", weights[layer].shape)
        # print("UPDATE exp_g_sq ", exp_g_sq[layer].shape)
        # print("UPDATE g_dict ", g_dict[layer].shape)
        g = g_dict[layer]
        exp_g_sq[layer] = decay_r * exp_g_sq[layer] + (1 - decay_r) * g ** 2

        weights[layer] += (learn_r * g) / (np.sqrt(exp_g_sq[layer] + eps))

        g_dict[layer] = np.zeros_like(weights[layer])


def relu(vector):
    # done
    vector[vector < 0] = 0
    return vector


def compute_gradient(gradient_log_p, hidden_layer_val, obs_val, weights):

    print("COMPUTE GRADIENT")
    # print("COMPUTE GRADIENT gradient_log_ ", gradient_log_p.shape)
    # print("COMPUTE GRADIENT hidden_layer ", hidden_layer_val.shape)
    # print("COMPUTE GRADIENT observation val: ", obs_val.shape)
    # print("COMPUTE GRADIENT weights 1: ", weights['1'].shape)
    # print("COMPUTE GRADIENT weights 2: ", weights['2'].shape)
    delta_L = gradient_log_p

    danny_delta = np.hstack([gradient_log_p,gradient_log_p,gradient_log_p])

    #dxC_dw2 = np.dot(hidden_layer_val.T, delta_L).ravel()
    dxC_dw2 = np.dot(hidden_layer_val.T,danny_delta)

    #print("Dc_w2 ", dxC_dw2.shape)
    #dxC_dw2 = dxC_dw2.reshape(dxC_dw2.shape[0],1)
    # print("Dc_w2 ", dxC_dw2.shape)

    #print("delta ", delta_L.shape)
    #delta_l2 = np.outer(delta_L, weights['2'])  # last layer, output layer

    delta_l2 = np.dot(danny_delta, weights['2'])
    #print("delta_l2 ", delta_l2.shape)
    delta_l2 = relu(delta_l2)
    dxC_dw1 = np.dot(delta_l2.T, obs_val)
    #print("obs_val " , obs_val.shape)
    #print("dxc_d1 ", dxC_dw1.shape)
    return {
        '1': dxC_dw1,
        '2': dxC_dw2.T
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