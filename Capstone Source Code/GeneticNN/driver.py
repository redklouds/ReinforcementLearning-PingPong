#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   driver.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import gym, random, threading, pickle
import numpy as np
from queue import Queue
from exitobject import ExitObject
from workthread import WorkThread
from helper_methods import generateChildren, mutate
from network import Network


envs = []
NUM_GENERATIONS = 50
NUM_NETWORKS = 10
PERCENT_NETS_TO_KILL = .25

INPUT_DIMENSION = 80*80

RESUME = True
#a 32bit pythopn teroperter only allows 2GB physical page siz ein ram
def main():
    pop = Queue() #population queue, storage for all our neural networks
    # initialize the Default starting networks

    if RESUME:
        for _ in range(2):
            f_name = "Hyper_param_P%s.p" % (_ + 1)
            f = open(f_name, "rb")
            _param = pickle.load(f)
            _n = Network(_param)
            pop.put(_n)

        for net in range(NUM_NETWORKS - 2):
            # create random networks
            # _net = Network(test_learning_rate, test_decay_rate,test_num_hidden_neuron,test_input_dimension)
            lr = round(random.random(), 2)
            hid_nur = random.randint(120, 800)
            dr = round(random.random(), 2)
            _w = {'1': np.random.randn(INPUT_DIMENSION, hid_nur) / np.sqrt(INPUT_DIMENSION),
                  '2': np.random.randn(hid_nur, 3) / np.sqrt(hid_nur)
                  }
            hyper_param = {'learning_rate': lr,
                           'num_hidden_neuron': hid_nur,
                           'decay_rate': dr,
                           'weights': _w
                           }

            _net = Network(hyper_param)
            pop.put(_net)

    else:
        for net in range(NUM_NETWORKS):
            # create random networks
            # _net = Network(test_learning_rate, test_decay_rate,test_num_hidden_neuron,test_input_dimension)
            lr = round(random.random(), 2)
            hid_nur = random.randint(120, 800)
            dr = round(random.random(), 2)
            _w = {'1': np.random.randn(INPUT_DIMENSION, hid_nur) / np.sqrt(INPUT_DIMENSION),
                  '2': np.random.randn(hid_nur, 3) / np.sqrt(hid_nur)
                  }
            hyper_param = {'learning_rate': lr,
                           'num_hidden_neuron': hid_nur,
                           'decay_rate': dr,
                           'weights': _w
                           }

            _net = Network(hyper_param)
            pop.put(_net)

    # prepare the enviroments
    for e in range(NUM_NETWORKS):
        # create a envrioment for each network
        envs.append(gym.make("Pong-v0"))

    # place where we put work that needs to be processed
    workQueue = Queue()
    # thread synchonize lock
    workQueueLock = threading.Lock()
    thExitController = ExitObject()
    # create our threads first
    threads = []
    for threadID in range(NUM_NETWORKS):
        th = WorkThread(threadID, thExitController, workQueueLock, workQueue)
        threads.append(th)
        th.start()

    for generation in range(NUM_GENERATIONS):
        # for each generation

        print("Starting Generation-%s" % generation)

        result_arr = []
        # starts our threads, gives each thread a network and an enviroment
        workQueueLock.acquire()
        print("Size of ppulation %s" % pop.qsize())
        print("Size of work queue %s" % workQueue.qsize())
        for i in range(NUM_NETWORKS):
            # get the lock when putting work into the work queue
            work = {
                'net': pop.get(),
                'env': envs[i],
                'result_array': result_arr
            }
            # put the work into the work queue, to let threads process them

            workQueue.put(work)
        workQueueLock.release()
        print("pop size: %s " % pop.qsize())
        # put work into the work queue again




        ############# hat program until all threads/games/networs ahve finished playing to compare scores
        while len(result_arr) != NUM_NETWORKS:
            pass
        print("the Networks have finshed playing...")

        ############# Pick top performers based on reward score
        parents = []
        p1 = 0
        for i in range(len(result_arr)):
            if result_arr[i].reward > result_arr[p1].reward:
                p1 = i
        parents.append(result_arr[p1])
        del result_arr[p1]

        p2 = 0
        for i in range(len(result_arr)):
            if result_arr[i].reward > result_arr[p2].reward:
                p2 = i
        parents.append(result_arr[p2])

        del result_arr[p2]
        file = open("Hyper_param_P1.p", "wb")
        print("saving parent 1 and parent 2 weights")
        pickle.dump(parents[0]._hyper_param, file)
        file1 = open("Hyper_param_P2.p", "wb")
        pickle.dump(parents[1]._hyper_param, file1)
        file.close()
        file1.close()

        print(len(result_arr))
        ############### pick the top performing parents

        amount_to_kill = int((PERCENT_NETS_TO_KILL * NUM_NETWORKS))
        children = generateChildren(parents, amount_to_kill)
        print("Number of Networks to kill %s" % amount_to_kill)
        ################# remove 10 from the population IE, result_array

        for i in range(amount_to_kill):
            to_remove = random.choice(result_arr)
            result_arr.remove(to_remove)

        ################ mutate the leftvoers that are not winners
        for _net in result_arr:
            mutate(_net)
        ################ After 10 networks killed off repopulate with new spawn

        result_arr.extend(children)  # merge existing Networks with chldren
        result_arr.extend(parents)
        for _net in result_arr:
            pop.put(_net)

            # now the remainer of the result_arr has the loswers , we need to mutate some of them and remove
            # however many we are thinking of breeding from the winners

            ################# we have the two winning parents, kill off 10(50%) and repopulate with spawn
    print("Finished program execution, stopping all threads ....")
    thExitController.setExit()
    print("Finished program execution, calling all sub threads to join main thread...")
    for th in threads:
        th.join()


main()
print("Finished ")


