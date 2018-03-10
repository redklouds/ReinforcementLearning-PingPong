import gym
import numpy as np
from queue import Queue
import threading
import pickle
#from threading import Queue
exitFlag = 0


envs = []
NUM_ENVIROMENTS = 5
NUM_GENERATIONS = 10
NUM_NETWORKS = 2






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


    def getAction(self, vector):
        action_idx = np.argmax(vector)
        if action_idx == 0:
            return 2 # go up
        elif action_idx == 1:
            return 0 # do nothing
        else:
            return 3 # go down




#clean up

prev = None
#prr_obs, prev_pre = preprocess_observation(obs, prev, 80*80)
#print(prr_obs.shape)
#result = _net.predict(prr_obs)

#print(result)


env = gym.make("Pong-v0")
env.reset()

file = open("ParentNetwork1.p","rb")
net = pickle.load(file)
file.close()
r = net.predict(np.random.randn(6400,1))
print(r)







