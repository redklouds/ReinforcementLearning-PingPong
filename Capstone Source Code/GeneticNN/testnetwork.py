

from multiprocessing import Process

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   testnetwork.py
# =#| Date:   3/10/2018
# =#|
# =#| Program Desc: This program is to test our top neural network's after they have been
# =#|                Selected and trained, against the computer, this program does not
# =#|                train the network just test the top performer from driver.py
# =#|     Usage:
# =#|
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import gym
import numpy as np
import pickle
from helper_methods import preprocess_observation
from network import Network

NUM_NETWORKS_TO_TEST  = 3# number of trained neural networks to test
def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3

def main(hyperParam):
    # load the network and the saved hyper parameters into this network
    _net = Network(hyper_param=hyperParam) # each function takes a hyper paramter dictionary
    env = gym.make("Pong-v0") # makes an enviroment
    obs = env.reset() # refreshes the enviroment
    prev_obs = None
    while True:
        #renders the environment visually
        env.render()
        #preprocess the 210 X 180 pixel frame image
        pro, prev_obs = preprocess_observation(obs, prev_obs, 80*80)
        #forward policy propagate through the current network
        a2,a1 = _net.predict1(pro)
        #suggest an action to perform in the environment
        action = choose_action(a1)
        #store the object frame pixels
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            #break

if __name__ == "__main__":

    print("Loading one network")

    f = open("HyperParam_obj.p","rb")
    network_params = pickle.load(f) # load the array with [network_param1, network_param2, ... network_paraN]

    #stop the program if the number of networks to test is out of bounds
    assert NUM_NETWORKS_TO_TEST <= len(network_params), "Not enough hyper parameters to test"
    for process in range(NUM_NETWORKS_TO_TEST):
        #grab an asscoiated network parameter from the parameter array and pass it to
        #the working method, each network MUST have its own process, openai gym DOES NOT allow
        # multiple threads from the same process to handle multiple environments
        p = Process(target = main, args=(network_params[process],))
        p.start()




