#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   network.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np

class Network:
    """
        {
            'learning_rate':int,
            'decay_rate':float,
            'num_hidden_neuron':int,
        }
    """

    def __init__(self, hyper_param=None):
        # def __init__(self, lr, dr, num_hidden_neurons, input_dimension):
        if hyper_param is None:
            self.setupDefaultParam()
        else:
            self._hyper_param = hyper_param  #
        self.reward = 0

    def setupDefaultParam(self):
        test_learning_rate = 0.7
        test_num_hidden_neurons = 200
        test_decay_rate = .98
        test_input_dimension = 80 * 80
        test_weights = {
            '1': np.random.randn(test_num_hidden_neurons, test_input_dimension) / np.sqrt(test_input_dimension),
            '2': np.random.randn(3, test_num_hidden_neurons) / np.sqrt(test_num_hidden_neurons)
            }
        self._hyper_param = {
            'learning_rate': test_learning_rate,
            'num_hidden_neuron': test_num_hidden_neurons,
            'decay_rate': test_decay_rate,
            'weights': test_weights
        }

    def getHyperParam(self):
        return self._hyper_param

    def ReLu(self, vector):
        vector[vector < 0] = 0
        return vector

    def reinitWeights(self):
        self._hyper_param['weights']['1'] = np.random.randn(80*80, self._hyper_param['num_hidden_neuron']) / np.sqrt(
            80 * 80)
        self._hyper_param['weights']['2'] = np.random.randn( self._hyper_param['num_hidden_neuron'], 3) / np.sqrt(
            self._hyper_param['num_hidden_neuron'])
    def softmax1(self,vector):
        # if(len(x.shape)==1):
        #  x = x[np.newaxis,...]
        probs = np.exp(vector - np.max(vector, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def softmax(self, vector):
        """Compute softmax values for each sets of scores in vector"""
        return np.exp(vector) / np.sum(np.exp(vector))

    def softmax3(self, v):
        x_exp = np.exp(v)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp/x_sum
        return s

    def setReward(self, r):
        self.reward = r


    def predict(self, input_p):
        n1 = input_p.dot(self.getHyperParam()['weights']['1'])

        #n1 = np.dot(self._hyper_param['weights']['1'], input_p)
        a1 = self.ReLu(n1)
        # Second layer
        n2 = a1.dot(self.getHyperParam()['weights']['2'])
        #n2 = np.dot(self._hyper_param['weights']['2'], a1)
        a2 = self.softmax1(n2)
        return a2, a1

    def getAction(self, vector):
        action_idx = np.argmax(vector)
        if action_idx == 0:
            return 2  # go up
        elif action_idx == 1:
            return 0  # do nothing
        else:
            return 3  # go down

    def __str__(self):
        return "LR: %s | #HidNur: %s | DR: %s " % (
        self._hyper_param['learning_rate'], self._hyper_param['num_hidden_neuron'], self._hyper_param['decay_rate'])
