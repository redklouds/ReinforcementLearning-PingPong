#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   network.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc: Network Object
#=#|
#=#| Usage: Used as a object to store our parameters for each neural network
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np
import random

class Network:
    """
    This is the Neural Network object, which stores the hyper parameters we are trying to
    Optimize, this Object contains forward prorogation methods, as well as activation functions
        {
            'learning_rate':int,
            'decay_rate':float,
            'num_hidden_neuron':int,
        }
    """

    def __init__(self, hyper_param=None, input_dimension=(80*80), name=None ) :
        self.reward = 0 #stored reward value, used in comparing a fitness function to this network
        self.input_dimension = input_dimension #input dimensions for this neural network
        if hyper_param is None:
            self.setupDefaultParam()
        else:
            self._hyper_param = hyper_param  #
        self.net_name = name

    def setupDefaultParam(self):
        """
        If network does not have a hyper parameter initialize this network with random values,
        the random values are the hyper parameters of this network.
        :return:None
        """

        lr = round(random.random(), 2) # learning_rate [0,1)
        dr = round(random.random(), 2) # decay_rate [0,1)
        hid_nur = random.randint(5, 400) #number hidden neurons [20, 700)

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
        assert type(dict()) == type(self._hyper_param['weights']), "Error hyper param, 'weights' initalize with wrong type, needs DICT GOT %s"% type(self._hyper_param['weights'])

        self._hyper_param['weights']['1'] = np.random.randn(num_hidden_nur, num_input_nur) / np.sqrt(num_hidden_nur)
        self._hyper_param['weights']['2'] = np.random.randn(num_hidden_nur) / np.sqrt(num_hidden_nur)

    def getHyperParam(self):
        """
        Getter for object hyper parameters
        :return: a dict of the hyper parameters stored within this object
        """
        return self._hyper_param

    def ReLu(self, vector):
        """
        ReLu activation function
        :param vector: Vector, Numpy
        :return: Vector
        """
        vector[vector < 0] = 0
        return vector

    def softmax1(self,vector):
        """
        Softmax activation function that uses, a log-softmax variant
        this is used if we have values that could potentially go out of bounds
        we esstentially noramlize our output
        :param vector:
        :return: Vector
        """
        # if(len(x.shape)==1):
        #  x = x[np.newaxis,...]
        probs = np.exp(vector - np.max(vector, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def softmax(self, vector):
        """Compute softmax values for each sets of scores in vector
        :param vector
        :return vector
        """
        return np.exp(vector) / np.sum(np.exp(vector))


    def setReward(self, r):
        self.reward = r
    def getReward(self):
        return self.reward

    def predict1(self, input_p ):
        """
        This is our forward propagation method, using a ReLu Activation Function, and a LogSigmoid Output activation function
        the last output will be a floating point value, in which a probability of an action to perform
        :param input_p: an 80*80 vector
        :return:
        """
        #first layer pass
        n1 = np.dot(self.getHyperParam()['weights']['1'], input_p)
        a1 = self.ReLu(n1)
        #second layer pass
        n2 = np.dot(a1, self.getHyperParam()['weights']['2'])
        a2 = self.sigmoid(n2)

        return n1, a2


    def predict(self, input_p):
        """
        This is our forward propagation method, using a ReLu activation function in the hidden layer, and a softmax activation
        Function for the output layer this function outputs 3 values of probability distribution, the higher of the values
        distinguish which action to most likely, or which class is most-likely to be choosen
        :param input_p:
        :return:
        """
        n1 = input_p.dot(self.getHyperParam()['weights']['1'])
        #n1 = np.dot(self._hyper_param['weights']['1'], input_p)
        a1 = self.ReLu(n1)
        # Second layer
        n2 = a1.dot(self.getHyperParam()['weights']['2'])
        #n2 = np.dot(self._hyper_param['weights']['2'], a1)
        a2 = self.softmax1(n2)
        return a2, a1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        #return 1/ (1 + expit(-x))

    def getAction(self, vector):
        action_idx = np.argmax(vector)
        if action_idx == 0:
            return 2  # go up
        elif action_idx == 1:
            return 0  # do nothing
        else:
            return 3  # go down

    def __str__(self):
        """
        To Print out our network , Overrides the toString Method
        :return: String
        """
        retString = "LR: %s | #HidNur: %s | DR: %s " % (
            self._hyper_param['learning_rate'], self._hyper_param['num_hidden_neuron'], self._hyper_param['decay_rate'])
        if self.net_name is not None:
            retString += "NAME %s " % self.net_name
        return retString