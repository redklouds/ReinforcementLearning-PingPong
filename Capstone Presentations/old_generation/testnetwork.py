

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

from old_generation.workthread import getAction


def main(hyperParam):
    # load the network and the saved hyper parameters into this network
    _net = Network(hyper_param=hyperParam)
    env = gym.make("Pong-v0")
    obs = env.reset()
    prev_obs = None
    while True:
        env.render()
        pro, prev_obs = preprocess_observation(obs, prev_obs, 80*80)
        a2,a1 = _net.predict(pro)
        u = np.random.uniform()
        print("a2 ", a2)
        prob_cum = np.cumsum(a2)
        a = np.where(u <= prob_cum)
        print("prob_cum " , prob_cum)
        action = getAction(a)
        print("actions : " ,action)
        obs, reward, done, info = env.step(action)
        if done:

            obs = env.reset()
            #break

if __name__ == "__main__":
    print("[+] Loading Hyper Parameters 1....")
    f = open("Hyper_param_P1.p", "rb")
    hyper_param = pickle.load(f)
    f.close()
    print("Loaded...")

    # print("[+] Loading Hyper Parameters 2....")
    # f = open("hyper_param_P2.p","rb")
    # hyper_param2 = pickle.load(f)
    # f.close()
    # print("Loaded....")


    p2 = Process(target =main, args=(hyper_param,) )
    #p3 = Process(target= main, args=(hyper_param2,) )
    p2.start()
    #p3.start()






