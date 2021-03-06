#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   helper_methods.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc: This is a group of multiple helper methods
#=#|
#=#| Usage: Assist in the program flow
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np
from network import Network
import random

def downsample(image):
    """
    reduce the resolution of the image
    """
    return image[::2, ::2, :]
def remove_color(image):
    """
    Removes all the Color of the raw image
    """
    return image[:, :, 0]

def remove_background(image):
    """
    :param image:
    :return:
    """
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def choose_action(probability):

    random_value = np.random.uniform()
    #print("probability" + str(random_value))
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3

def preprocess_observation(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195]  # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation

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
    """
    Standard Konnen Weight updating
    :param weights: Dict/Hash, weights of the network trying to update
    :param exp_g_sq: Numpy, Vector, of the gradients squared
    :param g_dict: Numpy,Vector, gradient values for each frame
    :param decay_r: Float, Decay Rate to be applied
    :param learn_r: Float, Learning Rate to be applied
    :return: None
    """
    eps = 1e-5
    for layer in weights.keys():
        g = g_dict[layer]
        exp_g_sq[layer] = decay_r * exp_g_sq[layer] + (1 - decay_r) * g ** 2

        weights[layer] += (learn_r * g) / (np.sqrt(exp_g_sq[layer] + eps))

        g_dict[layer] = np.zeros_like(weights[layer])
def ReLu(vector):
    """
    ReLu activation function
    :param vector: Vector, Numpy
    :return: Vector
    """
    vector[vector < 0] = 0
    return vector

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """
    Compute the gradient, using backprop given the reward/error values for each frame within
    the playing field
    :param gradient_log_p: Numpy, Vector Array
    :param hidden_layer_values:  Numpy, Matrix Array of the weights for the hidden layer for
    each frame
    :param observation_values: Numpy,Matrix each raw frame by frame values
    :param weights: Numpy, Matrix, the weights of the respective neural network
    :return: Dict of the gradients used to update the network
    """
    #propogate the output values+rewards+errors
    delta_L = gradient_log_p
    #propogate then with the hidden input layer
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    #find the cross entropy of the input values with the hidden outvalues
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = ReLu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
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
    """
    This method takes the gradient and applies a reward discount
    :param gradient_log: the gradient array/list, from action to expected value
    :param ep_rewards: the running frames reward array/list
    :param gamma: amount of gamma coefficient to discount to the reward
    :return: List/Array of the newly discounted reward array with gradient constant(multiplied into the reward, used for
    our gradient backprop to learn)
    """
    discount_ep_reward = discount_reward(ep_rewards, gamma) # make an array that applies a discount to actions that are
    # more relevant,
    discount_ep_reward -= np.mean(discount_ep_reward) # normalize this reward vector
    discount_ep_reward /= np.std(discount_ep_reward)
    return gradient_log * discount_ep_reward # apply the gradient score to the respective rewards, this tells us the gradient(error to learn from)

def mutate(network):
    """
    randomly select a parameter to change and randomize that parameter, may need to change more than 1 parameter at a time
    :param network: Network Object
    :return: None
    """

    for param in network.getHyperParam().keys():
        #for each hyper parameter lets introduce some randomization
        if param == 'learning_rate' or param == 'decay_rate':
            network.getHyperParam()[param] = random.random()
        elif param == 'weights' or param == 'num_hidden_neuron':
            # if we run into hidden layer change OR weight change we need to reinitialize the weights , because the dimensions will need to be updated
            network.randomizeWeights(random.randint(5, 400), 80 * 80)

def generateChildren(parents,num_children):
    """
    return a list of networks that are a combinations of the array of parents given, using
    random selection process
    :param parents: Neural networks in which we want to inherit properties (parameters) from
    :param num_children: the number of spawm we want to generate
    :return: an array/list of spawn
    """
    print("Calling Generating children")
    spawn = []
    for i in range(num_children):
        _param = {} # temp hyper parameters for the child network
        for param in parents[0].getHyperParam().keys():
            #for each paramter in the parent, choose which parent attribute to inherit from
            _param[param] = random.choice(parents).getHyperParam()[param]
        #create the child network
        _net = Network(_param)
        #add it to the running children population
        spawn.append(_net)
    return spawn