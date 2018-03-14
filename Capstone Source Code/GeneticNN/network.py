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
import random
class Network:
    """
        {
            'learning_rate':int,
            'decay_rate':float,
            'num_hidden_neuron':int,
        }
    """

    def __init__(self, hyper_param=None, input_dimension= (80*80) ) :
        self.reward = 0 #stored reward value, used in comparing a fitness function to this network
        self.input_dimension = input_dimension #input dimensions for this neural network
        if hyper_param is None:
            self.setupDefaultParam()
        else:
            self._hyper_param = hyper_param  #

    def setupDefaultParam(self):
        """
        If network does not have a hyper parameter initialize this network with random values
        :return:
        """

        lr = round(random.random(), 2) # learning_rate [0,1)
        dr = round(random.random(), 2) # decay_rate [0,1)
        hid_nur = random.randint(20, 700) #number hidden neurons [20, 700)

        _w = {}
        self._hyper_param = {'learning_rate': lr,
                       'num_hidden_neuron': hid_nur,
                       'decay_rate': dr,
                       'weights': _w
                       }
        self.randomizeWeights(hid_nur, self.input_dimension)

    def randomizeWeights(self, num_hidden_nur, num_input_nur):
        """
        PRECONDITION: network must be initialized with a hyper-parameter 'weights' hash/dict prior to calling
        this method
        :param num_hidden_nur: number of hidden neurons
        :param num_input_nur: number of input neurons
        :return: None
        """
        assert 'weights' in self._hyper_param.keys(), "ERROR with object initalization, missing 'weights' hyper param"
        assert type(dict) == type(self._hyper_param['weights']), "Error hyper param, 'weights' initalize with wrong type, needs DICT"

        self._hyper_param['weights']['1'] = np.random.randn(num_hidden_nur, num_input_nur) / np.sqrt(num_hidden_nur)
        self._hyper_param['weights']['2'] = np.random.randn(num_hidden_nur) / np.sqrt(num_hidden_nur)

    def getHyperParam(self):
        return self._hyper_param

    def ReLu(self, vector):
        vector[vector < 0] = 0
        return vector

    def reinitWeights(self):
        self._hyper_param['weights']['1'] = np.random.randn(self._hyper_param['num_hidden_neuron'],80*80) / np.sqrt(
            80 * 80)
        self._hyper_param['weights']['2'] = np.random.randn( self._hyper_param['num_hidden_neuron']) / np.sqrt(
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
    def relu(self,v):
        v[v < 0] =0
        return v

    def predict1(self, input_p ):
        hidden_val = np.dot(self.getHyperParam()['weights']['1'], input_p)
        hidden_val = self.relu(hidden_val)
        out_val = np.dot(hidden_val, self.getHyperParam()['weights']['2'])
        out_val = self.sigmoid(out_val)
        return hidden_val, out_val


    def predict(self, input_p):
        n1 = input_p.dot(self.getHyperParam()['weights']['1'])

        #n1 = np.dot(self._hyper_param['weights']['1'], input_p)
        a1 = self.ReLu(n1)
        # Second layer
        n2 = a1.dot(self.getHyperParam()['weights']['2'])
        #n2 = np.dot(self._hyper_param['weights']['2'], a1)
        a2 = self.softmax1(n2)
        return a2, a1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
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
