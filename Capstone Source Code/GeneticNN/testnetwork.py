

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   testnetwork.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc: This program is to test our top neural network's after they have been
#=#|                Selected and trained, against the computer, this program does not
#=#|                train the network just test the top performer from driver.py
#=#|     Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import gym, pickle
from helper_methods import preprocess_observation
from network import Network
from multiprocessing import Process




def main(hyperParam):
    # load the network and the saved hyper parameters into this network
    _net = Network(hyper_param=hyperParam)
    env = gym.make("Pong-v0")
    obs = env.reset()
    prev_obs = None
    while True:
        env.render()
        pro, prev_obs = preprocess_observation(obs, prev_obs, 80*80)
        a1_output,a2_output = _net.predict(pro)
        action = _net.getAction(a2_output)
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

    print("[+] Loading Hyper Parameters 2....")
    f = open("hyper_param_P2.p","rb")
    hyper_param2 = pickle.load(f)
    f.close()
    print("Loaded....")


    p2 = Process(target =main, args=(hyper_param,) )
    p3 = Process(target= main, args=(hyper_param2,) )
    p2.start()
    p3.start()






