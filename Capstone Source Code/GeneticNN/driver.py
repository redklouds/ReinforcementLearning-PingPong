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
from queue import Queue #Work Queue, and Population Queue
import gym, pickle, random, threading #openai Environments, save weight state, random generator, threading support
from network import Network # neural network object
from exitobject import ExitObject # threaded exit object
from helper_methods import generateChildren, mutate
import numpy as np
from workthread import WorkThread
NUM_GENERATIONS = 800
NUM_NETWORKS = 15
PERCENT_NETS_TO_KILL = .45
NUM_PARENTS = 3
INPUT_DIMENSION = 80*80
RESUME = True
INCLUDE_CAPARSION = True
NUM_ROUNDS = 10 #value determines how many rounds of ping-pong out of 21(is a round) do we play until we update our weights
# and return to a next generation, the more we train the longer, but the better results we will have from a good gradient batch to learn from



def makeEnvironments(num_env, list):
    """
    Helper to initialize the number of given  environments
    num_env: number of environments to initialize
    list: populate the list with these environments
    """
    for env in range(num_env):
        list.append(gym.make("Pong-v0"))

def makeRandomNetworks(input_dimensions,num_networks, queue):
    """
    makes the number of random Neural Networks with random parameters, and puts them into the given
    Queue

    """
    for net in range(num_networks):
        # create random network
        lr = round(random.random(), 2)
        hid_nur = random.randint(120, 800)
        dr = round(random.random(), 2)
        _w = {'1': np.random.randn(hid_nur, input_dimensions) / np.sqrt(input_dimensions),
              '2': np.random.randn(hid_nur) / np.sqrt(hid_nur)
              }
        hyper_param = {'learning_rate': lr,
                       'num_hidden_neuron': hid_nur,
                       'decay_rate': dr,
                       'weights': _w
                       }

        _net = Network(hyper_param)
        queue.put(_net)


def makeThreads(num_threads, thread_storage, thread_exit_obj, thread_work_que_lock, thread_work_que , threadPrefix = None):
    """
    Makes the number of threads and starts them
    num_thread: number threads to make and start
    thread_storage: pop the threads into this array
    thread_exit_obj: exist object that signals the threads to quit working
    thread_work_que_lock: the mutex lock object for all threads to use to synchronize
    thread_work_que: the queue that the threads get their work from
    """
    for threadID in range(num_threads):
        threadName = threadID
        if threadPrefix is not None:
            threadName = "%s %s" %(threadPrefix, threadID)
        th = WorkThread(threadName, thread_exit_obj, thread_work_que_lock, thread_work_que)
        thread_storage.append(th)
        th.start()



#a 32bit python Interperter only allows 2GB physical page siz ein ram
def main():
    environment_list = []
    pop = Queue() #population queue, storage for all our neural networks
    # initialize the Default starting networks

    if RESUME:
        # RESUME Where we left off all, meaning re-introduce the configurations for the winning parents of the last generation
        # then fill in the missing amount with random new networks, in theory the winning parents should still stay on top in the following
        # generation.
        file = open("HyperParam_obj.p", "rb")
        data = pickle.load(file)
        assert NUM_PARENTS <= len(data), "Not enough saved parent networks, please check NUM_PARENTS PARAMTER"
        file.close()

        for _param in range(NUM_PARENTS):
            _n = Network(data[_param])
            pop.put(_n)


        makeRandomNetworks( INPUT_DIMENSION, (NUM_NETWORKS - NUM_PARENTS ) , pop )

    else:
        makeRandomNetworks(INPUT_DIMENSION, NUM_NETWORKS, pop)

    makeEnvironments(NUM_NETWORKS, environment_list)
    # place where we put work that needs to be processed
    workQueue = Queue()
    # thread synchronize lock
    workQueueLock = threading.Lock()
    thExitController = ExitObject()
    # create our threads first
    threads = []



    #number of networks ot make, the thread_list, the exit object, synchronized lock, and the work Queue
    makeThreads(NUM_NETWORKS , threads,thExitController,workQueueLock, workQueue)


    ##############################################################################################
    ###################################################################

    if INCLUDE_CAPARSION:
        __env = gym.make("Pong-v0")
        makeThreads(1, threads, thExitController,workQueueLock,workQueue, "TRTR" )

        f = open("comparison.p", "rb")
        a = pickle.load(f)
        f.close()
        __net = Network(a, name="COMPARSION")
        __work = {
            'net': __net,
            'env': __env,
            'tag': True,
            'num_rounds': NUM_ROUNDS,
        }
        workQueueLock.acquire()
        workQueue.put(__work)
        workQueueLock.release()


    ########################################################################
    #########################################################################################
    for generation in range(NUM_GENERATIONS):
        result_arr = []

        print("[+] Starting Generation-%s" % generation)
        print("[+] Size of out population %s" % pop.qsize())
        print("[+] Checking... Size of work queue %s" % workQueue.qsize())

        # grab our lock so we can synchronize the workQueue and prevent any unwanted threads from corrupting the data
        workQueueLock.acquire()
        for i in range(NUM_NETWORKS):
            # create a work data structure,
            # each work will contain the network, environment, and the return array that associated network
            # will be using
            work = {
                'net': pop.get(),
                'env': environment_list[i],
                'result_array': result_arr,
                'num_rounds' : NUM_ROUNDS
            }
            # put the work into the work queue, to let threads process them
            workQueue.put(work)
        #after all work structures are placed within the workQueue we release the thread locks
        workQueueLock.release()
        print("[+] Checking.... pop size after work loaded: %s " % pop.qsize())

        ############# Pause Program Counter of the main thread. ##############################

        # at this point all threads are enQueuing work from the work Queue and running their respective matches
        # individually, so as long as our resulting array, which each thread will put the trained network back into
        # is not equal to the size of our expected networks, we will pause the program counter here.
        while len(result_arr) != NUM_NETWORKS:
            pass
        print("[+} all Networks have finished playing...")

        ############# Pick top performers based on reward score ######################

        # go into the leader board and get the top number of NUM_PARENTS
        assert not NUM_PARENTS > NUM_NETWORKS, "Error NUM_PARENTS parameter is out of bounds"
        parents = []
        #picks the top contenders
        for p in range(NUM_PARENTS):
            #get the parent index of the current winning network
            parent_index = 0
            for i in range(len(result_arr)):
                #search for the next largest winner
                if result_arr[i].getReward() > result_arr[parent_index].getReward():
                    parent_index = i
            parents.append(result_arr[parent_index])# add this network to the parents population
            del result_arr[parent_index]# remove the parent from the population(general)


        #################################################################
        #
        #           Leader board, Fitness Assessment
        #
        #################################################################
        winning_param = []
        #at the end of each generation we save the winning parents in a file
        for _network_param in parents:
            # create an object that saves the current parents hyper parameter(to use for testing performance)
            winning_param.append(_network_param.getHyperParam())
        # save an array of parameters of the winning networks, in the event we want to test , or re train with a different configuration
        file = open("HyperParam_obj.p","wb")
        pickle.dump(winning_param, file)
        file.close()


        ##################################################################
        #
        #           Breeding the top contenders
        #
        ###################################################################
        # this is the breeding stage, calculate the number of networks to kill from the population
        amount_to_kill = int((PERCENT_NETS_TO_KILL * NUM_NETWORKS))
        # call generate children to breed from the parents, returning the number of next generation of children networks
        children = generateChildren(parents, amount_to_kill)

        print("Number of Networks to kill %s" % amount_to_kill)


        for i in range(amount_to_kill):
            # randomly select and remove the amount of networks to kill from the population
            to_remove = random.choice(result_arr)
            result_arr.remove(to_remove)

        ######################################################################
        #
        #           Mutating the left over networks
        #
        #######################################################################

        # After removing the number of networks specified, the remaining networks in the population
        # will need to be mutated, to introduce outside variance
        for _net in result_arr:
            mutate(_net)


        #merge the new generation with th population
        result_arr.extend(children)
        #merge the parents into the generation pool as well, the children may not always be better than the parents
        result_arr.extend(parents)



        #######################################################################
        #
        #       Next Generation - Mutated Networks,
        #                           Breaded children networks, Previous winning networks
        #                           Back into the population
        #       **** Completes this Generation  ****
        #######################################################################
        for _net in result_arr:
            pop.put(_net)




    #############################
    #   End of Training, all Generations have been finished
    #
    ############################
    print("Finished program execution, stopping all threads ....")
    #signal the threads to stop
    thExitController.setExit()
    print("Finished program execution, calling all sub threads to join main thread...")
    # telling all threads to join the main thread.
    for th in threads:
        th.join()

#### Entry point of the program
main()
print("Finished ")


