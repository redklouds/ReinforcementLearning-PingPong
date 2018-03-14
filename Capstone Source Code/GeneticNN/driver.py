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
from queue import Queue
import gym
import numpy as np
import pickle
import random
import threading
from network import Network
from workthread import WorkThread
from exitobject import ExitObject
envs = []
NUM_GENERATIONS = 600
NUM_NETWORKS = 20
PERCENT_NETS_TO_KILL = .45
NUM_PARENTS = 5
INPUT_DIMENSION = 80*80
RESUME = True


def generateRandomNetworks(num_networks, queue):
    """ makes the number of random Neural Networks with random parameters, and puts them into the given
    Queue"""
    for net in range(num_networks):
        # create random network
        lr = round(random.random(), 2)
        hid_nur = random.randint(120, 800)
        dr = round(random.random(), 2)
        _w = {'1': np.random.randn(hid_nur, INPUT_DIMENSION) / np.sqrt(INPUT_DIMENSION),
              '2': np.random.randn(hid_nur) / np.sqrt(hid_nur)
              }
        hyper_param = {'learning_rate': lr,
                       'num_hidden_neuron': hid_nur,
                       'decay_rate': dr,
                       'weights': _w
                       }

        _net = Network(hyper_param)
        queue.put(_net)

def makeThreads(num_threads, thread_storage, thread_exit_obj, thread_work_que_lock, thread_work_que ):
    """
    Makes the number of threads and starts them
    num_thread: number threads to make and start
    thread_storage: pop the threads into this array
    thread_exit_obj: exist object that signals the threads to quit working
    thread_work_que_lock: the mutex lock object for all threads to use to synchronize
    thread_work_que: the queue that the threads get their work from
    """
    for threadID in range(num_threads):
        th = WorkThread(threadID, thread_exit_obj, thread_work_que_lock, thread_work_que)
        thread_storage.append(th)
        th.start()

def makeEnvironments(num_env, list):
    """Helper to initialize the number of given  environments"""
    for env in range(num_env):
        list.append(gym.make("Pong-v0"))


#a 32bit pythopn teroperter only allows 2GB physical page siz ein ram
def main():
    pop = Queue() #population queue, storage for all our neural networks
    # initialize the Default starting networks

    if RESUME:
        # RESUME Where we left off all, meaning re-introduce the configurations for the winning parents of the last generation
        # then fill in the missing amount with random new networks, in theory the winning parents should still stay on top in the following
        # generation.
        file = open("HyperParam_obj.p", "rb")
        data = pickle.load(file)
        file.close()

        for _param in data:
            _n = Network(_param)
            pop.put(_n)
        # for resumed_network in range(2):
        #     f_name = "Hyper_param_P%s.p" % (_ + 1)
        #     f = open(f_name, "rb")
        #     _param = pickle.load(f)
        #     _n = Network(_param)
        #     pop.put(_n)


        generateRandomNetworks( (NUM_NETWORKS - len(data) ) , pop )
        # for net in range(NUM_NETWORKS - 2):
        #     # create random networks
        #     # _net = Network(test_learning_rate, test_decay_rate,test_num_hidden_neuron,test_input_dimension)
        #     lr = round(random.random(), 2)
        #     hid_nur = random.randint(120, 800)
        #     dr = round(random.random(), 2)
        #     _w = {'1': np.random.randn(hid_nur, INPUT_DIMENSION) / np.sqrt(INPUT_DIMENSION),
        #           '2': np.random.randn(hid_nur) / np.sqrt(hid_nur)
        #           }
        #     hyper_param = {'learning_rate': lr,
        #                    'num_hidden_neuron': hid_nur,
        #                    'decay_rate': dr,
        #                    'weights': _w
        #                    }
        #
        #     _net = Network(hyper_param)
        #     pop.put(_net)

    else:
        generateRandomNetworks(NUM_NETWORKS, pop)
        # for net in range(NUM_NETWORKS):
        #     # create random networks
        #     # _net = Network(test_learning_rate, test_decay_rate,test_num_hidden_neuron,test_input_dimension)
        #     lr = round(random.random(), 2)
        #     hid_nur = random.randint(120, 800)
        #     dr = round(random.random(), 2)
        #     _w = {'1': np.random.randn(hid_nur, INPUT_DIMENSION) / np.sqrt(INPUT_DIMENSION),
        #           '2': np.random.randn(hid_nur) / np.sqrt(hid_nur)
        #           }
        #     hyper_param = {'learning_rate': lr,
        #                    'num_hidden_neuron': hid_nur,
        #                    'decay_rate': dr,
        #                    'weights': _w
        #                    }
        #
        #     _net = Network(hyper_param)
        #     pop.put(_net)

    # # prepare the environments
    # for e in range(NUM_NETWORKS):
    #     # create a environments for each network
    #     envs.append(gym.make("Pong-v0"))
    #
    makeEnvironments(NUM_NETWORKS, envs)
    # place where we put work that needs to be processed
    workQueue = Queue()
    # thread synchronize lock
    workQueueLock = threading.Lock()
    thExitController = ExitObject()
    # create our threads first
    threads = []

    makeThreads(NUM_NETWORKS, threads,thExitController,workQueueLock, workQueue)
    # for threadID in range(NUM_NETWORKS):
    #     th = WorkThread(threadID, thExitController, workQueueLock, workQueue)
    #     threads.append(th)
    #     th.start()

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
        #picks the top contenders
        for p in range(NUM_PARENTS):
            #get the parent index of the current winning network
            parent_index = 0
            for i in range(len(result_arr)):
                #search for the next largest winner
                if result_arr[i] > result_arr[parent_index]:
                    parent_index = i
            parents.append(result_arr[parent_index])# add this network to the parents population
            del result_arr[parent_index]# remove the parent from the population(general)


        # collecting parameters of winning network
        winning_param = []
        #at the end of each generation we save the winning parents in a file
        for _network_param in parents:
            # create an object that saves the current parents hyper parameter(to use for testing performance)
            winning_param.append(_network_param.getHyperParam())

        file = open("HyperParam_obj.p","wb")
        pickle.dump(file, winning_param)
        file.close()

        ############### pick the top performing parents

        amount_to_kill = int((PERCENT_NETS_TO_KILL * NUM_NETWORKS))
        children = generateChildren(parents, amount_to_kill)
        print("Number of Networks to kill %s" % amount_to_kill)
        ################# remove 10 from the population IE, result_array

        for i in range(amount_to_kill):
            to_remove = random.choice(result_arr)
            result_arr.remove(to_remove)

        #the result_arr will have the remaining population that has been mutated
        for _net in result_arr:
            mutate(_net)
        ################ After 10 networks killed off repopulate with new spawn
        #merge the new generation with th population
        result_arr.extend(children)
        #merge the parents into the generation pool as well, the children may not always be better than the parents
        result_arr.extend(parents)

        #populate our population Work queue for the threads
        for _net in result_arr:
            pop.put(_net)

    #all generations have finished
    print("Finished program execution, stopping all threads ....")
    #signal the threads to stop
    thExitController.setExit()
    print("Finished program execution, calling all sub threads to join main thread...")
    for th in threads:
        th.join()


main()
print("Finished ")


