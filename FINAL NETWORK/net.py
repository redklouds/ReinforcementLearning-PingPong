import gym
import numpy as np
from queue import Queue
import threading
import pickle
import random
from network import Network
envs = []
NUM_ENVIROMENTS = 5
NUM_GENERATIONS = 15
NUM_NETWORKS = 12
PERCENT_NETS_TO_KILL = .50


class ExitObject:
    ## checkl
    """Threads exit object, used for calling all threads to stop queueing for work"""
    def __init__(self, exit=False):
        self.exit = exit
    def setExit(self):
        self.exit = True


class WorkThread(threading.Thread):
    #still needs works
    def __init__(self,tID, exit, lock, workQueue):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.exitFlag = exit
        self.QueueLock = lock
        self.work_queue = workQueue
    
    def run(self):
        print("Thread- %s, has been started.. waiting for work" % self.threadID)
        while not self.exitFlag.exit:
            #print("Thread-%s trying to get work%s" %(self.threadID, self.work_queue.qsize()))
            #print("Thread-%s waiting for lock..." % self.threadID)
            
            self.QueueLock.acquire()
            #print("Thread-%s got the LOCK!" % self.threadID)
            if not self.work_queue.empty():
                work = self.work_queue.get()
#got the work ,{'net','env','result_array'}
                self.QueueLock.release()
                print("Thread-%s has network: %s" %(self.threadID, work['net'].__str__()))
                self.doWork(work['net'],work['env'],work['result_array'])
            else:
                self.QueueLock.release()
    






    def doWork(self, network, enviroment, result_array):
        print("Thread-%s doing work..." % self.threadID)
        obs = enviroment.reset()
        reward_sum = 0
        prev_processed_obser= None
        running_reward = None
        
        exp_gradient_squared = {}
        gradient_dict = {}
        gamma = .99
        
        for layer in network.getHyperParam()['weights'].keys():
            #basically populating the expected Graident values, used for learning later on
            #populate the gradient squared, with all weights in all alyers
            exp_gradient_squared[layer] = np.zeros_like(network.getHyperParam()['weights'][layer])
            gradient_dict[layer] = np.zeros_like(network.getHyperParam()['weights'][layer])
        
        ep_hidden_layer_vals = []
        ep_obs = []
        ep_gradient_log_ps = []
        ep_rewards = []
        
        while True:
            #self.QueueLock.acquire()
            #enviroment.render()
            #self.QueueLock.release()
            #print("Thread-%s WOrking and rendering" % self.threadID)
            processed_obs, prev_processed_obser = preprocess_observation(obs, prev_processed_obser, 80*80)

            #hidden_lay_values, action = network.predict(

            a1,a2 = network.predict(processed_obs)
            #print("result: %s" %result)
            
            
            #ep_obs.append(np.reshape(processed_obs, (processed_obs.shape[0],1)).T)
            #print("Hidden alyer size",a1.shape)
            
            #print("OBERSAVATIONS SIZE" , processed_obs.shape)
            #ep_hidden_layer_vals.append(np.reshape(a1, (a1.shape[0],1)).T)
        
        
            ep_obs.append(processed_obs)
            
            ep_hidden_layer_vals.append(a1)
            

            action  = network.getAction(a2)
            






            obs, reward, done, info = enviroment.step(action)
            
            #print("action: %s " % action)
                #print("reward: %s reward sum: %s" % (reward,reward_sum))
            reward_sum += reward
            
            ep_rewards.append(reward)
            
            #print("thread-%s reward: %s | done: %s | " %( self.threadID, reward, done))
            if(reward > 0.0):
                print("REWARDED %s" % reward)
        
            fk_label = np.argmax(a2) # a2 is a vector, get the index to repersent our label, 0, 1,2
            loss_function_gradient = fk_label - a2[fk_label]
            #print("Loss function gradient : %s "  % loss_function_gradient)
            
            ep_gradient_log_ps.append(loss_function_gradient)
            if done:
                #episode_numer += 1
                #print("Dont playing around: %s" % reward_sum)
                
                #form a single entity for all our recorded values for this game
                del ep_hidden_layer_vals[0]
                ep_hidden_layer_vals = np.vstack(ep_hidden_layer_vals)
                del ep_obs[0]
                ep_obs = np.vstack(ep_obs)
          
                del ep_gradient_log_ps[0]
                ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
                del ep_rewards[0]
                ep_rewards = np.vstack(ep_rewards)
                
                #change the gradient of the log_ps based on discount rewards
                
                ep_gradient_log_ps_discounted = discount_with_rewards(ep_gradient_log_ps, ep_rewards, gamma)
                

                gradient = compute_gradient( ep_gradient_log_ps_discounted,
                                            ep_hidden_layer_vals,
                                            ep_obs,
                                            network.getHyperParam()['weights'])
                            
                #print("gradient size 1", gradient['1'].shape)
                #print("gradient_dict size 1", gradient_dict['1'].shape)
#               print("gradient size 2 size ", gradient['2'].shape)
#               print("grandient_dict 2 size",gradient_dict['2'].shape)
#               for layer in gradient:
                #    gradient_dict[layer] += gradient[layer]

                #gradient_dict['1'] = gradient['1']
                #gradient_dict['2'] = gradient['2']
                #exp_gradient_squared['1'] = gradient['1']
                #exp_gradient_squared['2'] = gradient['2']



                #non batch updates
                update_weights(network.getHyperParam()['weights'], exp_gradient_squared, gradient_dict, network.getHyperParam()['decay_rate'], network.getHyperParam()['learning_rate'])
                                            
                print("thread-%s has finished playing... reward is: %s" % (self.threadID, reward_sum))
                #put the network back into the result array
                network.reward = reward_sum
                self.QueueLock.acquire()
                result_array.append(network)
                self.QueueLock.release()
                
                prev_processed_obser = None
                break



def downsample(image):
    #done
    return image[::2, ::2, :]
def remove_color(image):
    #done
    """ upon inspection the third dimension is the RGB value"""
    return image[:, :, 0]
def remove_background(image):
    #done
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def preprocess_observation(input_obs, pre_processed_obs, input_dimension):
    #done
    """Remove the score from the input,  """
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



def update_weights(weights, exp_g_sq, g_dict, decay_r, learn_r):
    #done
    eps = 1e-5
    for layer in weights.keys():
        g = g_dict[layer]
        exp_g_sq[layer] = decay_r * exp_g_sq[layer] + (1-decay_r) * g **2
        print("Weights :" , weights[layer].shape)
        print("g : " ,g.shape)
        print("exp_g_sqrt: ", exp_g_sq[layer].shape)
        weights[layer] += (learn_r * g) / (np.sqrt(exp_g_sq[layer] + eps))

        g_dict[layer] = np.zeros_like(weights[layer])

def relu(vector):
    #done
    vector[vector < 0] = 0
    return vector

def compute_gradient(gradient_log_p, hidden_layer_val, obs_val, weights):
  
    delta_L = gradient_log_p
 
    dxC_dw2 = np.dot(hidden_layer_val.T, delta_L).ravel()
    print("delta_L : ",delta_L.shape)
    print("dxC_dw2: " , dxC_dw2.shape)
    print("Layer 2 Weights : ", weights['2'].shape)
    
    delta_l2 = np.outer(delta_L, weights['2']) #last layer, output layer
    print("delta_l2: ",delta_l2.shape)
    delta_l2 = relu(delta_l2)
    dxC_dw1 = np.dot(delta_l2.T, obs_val)
    print("dxC_dw1: " ,dxC_dw1.shape)
    return {
        '1': dxC_dw1,
        '2': dxC_dw2
    }

def discount_reward(reward, gamma):
    """Discount ensures that actions that were taken in the beginning weight less of an
        importance to the result than actions that were taken 2 steps prior to this result"""
    discount_r = np.zeros_like(reward)
    r_total = 0
    for _ in reversed(range(0, reward.size)):
        if reward[_] != 0:
            r_total = 0
        r_total = r_total * gamma + reward[_]
        discount_r[_] = r_total
    return discount_r

def discount_with_rewards(gradient_log, ep_rewards, gamma):
    discount_ep_reward = discount_reward(ep_rewards, gamma)
    
    discount_ep_reward -= np.mean(discount_ep_reward)
    discount_ep_reward /= np.std(discount_ep_reward)
    return gradient_log * discount_ep_reward

class Network():
    """
        {
            'learning_rate':int,
            'decay_rate':float,
            'num_hidden_neuron':int,
        }
    """
    def __init__(self, hyper_param = None):
        #def __init__(self, lr, dr, num_hidden_neurons, input_dimension):
        if hyper_param is None:
            self.setupDefaultParam()
        else:
            self._hyper_param = hyper_param #
        self.reward = 0

    def setupDefaultParam(self):
        test_learning_rate = 0.7
        test_num_hidden_neurons = 200
        test_decay_rate = .98
        test_input_dimension = 80*80
        test_weights = {'1':np.random.randn(test_num_hidden_neurons,test_input_dimension)/ np.sqrt(test_input_dimension),
        '2':np.random.randn(3,test_num_hidden_neurons) /np.sqrt(test_num_hidden_neurons)
        }
        self._hyper_param = {
                'learning_rate': test_learning_rate,
                'num_hidden_neuron': test_num_hidden_neurons,
                'decay_rate':test_decay_rate,
                'weights': test_weights
            }

    def getHyperParam(self):
        return self._hyper_param
    def ReLu(self, vector):
        vector[vector < 0] = 0
        return vector
    def reinitWeights(self):
        self._hyper_param['weights']['1'] = np.random.randn(self._hyper_param['num_hidden_neuron'], 80*80) / np.sqrt(80*80)
        self._hyper_param['weights']['2'] = np.random.randn(3, self._hyper_param['num_hidden_neuron']) / np.sqrt(self._hyper_param['num_hidden_neuron'])
    
    def softmax(self,vector):
        """Compute softmax values for each sets of scores in vector"""
        return np.exp(vector) / np.sum(np.exp(vector), axis=0)
    
    def setReward(self, r):
        self.reward = r
    
    def predict(self, input_p):
        n1 = np.dot(self._hyper_param['weights']['1'],input_p)
        a1 = self.ReLu(n1)
        #Second layer
        n2 = np.dot(self._hyper_param['weights']['2'],a1)
        a2 = self.softmax(n2)
        return a1,a2


    def getAction(self, vector):
        action_idx = np.argmax(vector)
        if action_idx == 0:
            return 2 # go up
        elif action_idx == 1:
            return 0 # do nothing
        else:
            return 3 # go down
    def __str__(self):
        return "LR: %s | #HidNur: %s | DR: %s " % (self._hyper_param['learning_rate'], self._hyper_param['num_hidden_neuron'],self._hyper_param['decay_rate'])

def mutate(network):
    #done
    #given a network randomly choose a parameter and randomize that parameter
    param_to_change = random.choice(list(network.getHyperParam().keys()))
    if param_to_change == 'learning_rate' or param_to_change == 'decay_rate':
        network.getHyperParam()[param_to_change] = random.random()
    elif param_to_change == 'weights' or param_to_change == 'num_hidden_neuron':
#if we run into hidden layer change OR weight change we need to reinitatlize the weights , because the dimensions will need to be updated
        network.getHyperParam()['num_hidden_neuron'] = random.randint(120, 700)
        network.reinitWeights() #after changing the number of hidden neurons we need to reintalze the shape of our weights

def generateChildren(parents,num_children):
    #done
    spawn = []
    for i in range(num_children):
        _param = {}
        for param in parents[0].getHyperParam().keys():
            #if param == 'weights':
            #    _param['weights'] = {}
            #    print("comparing weights %s" % param)
            #    for w in parents[0].getHyperParam()[param].keys():
            #        _param[param][w] = random.choice(parents).getHyperParam()[param][w]
            #else:
            #    _param[param] = random.choice(parents).getHyperParam()[param]
            _param[param] = random.choice(parents).getHyperParam()[param]
        _net = Network(_param)
        spawn.append(_net)
    return spawn

def main():
    #done
    pop = Queue()
    #initalize the Default starting networks
    test_learning_rate = 0.7
    test_num_hidden_neurons = 200
    test_decay_rate = .98
    test_input_dimension = 80*80
    test_weights = {'1':np.random.randn(test_num_hidden_neurons,test_input_dimension)/ np.sqrt(test_input_dimension),
        '2':np.random.randn(3,test_num_hidden_neurons) /np.sqrt(test_num_hidden_neurons)
    }
    test_hyper_param = {
                        'learning_rate': test_learning_rate,
                        'num_hidden_neuron': test_num_hidden_neurons,
                        'decay_rate':test_decay_rate,
                        'weights': test_weights
                        }

    import random
    for net in range(NUM_NETWORKS):
        #create random networks
        #_net = Network(test_learning_rate, test_decay_rate,test_num_hidden_neuron,test_input_dimension)
        lr = round(random.random(),2)
        hid_nur = random.randint(120,600)
        dr = round(random.random(),2)
        _w = {'1': np.random.randn(hid_nur, 80*80) / np.sqrt(80*80),
            '2': np.random.randn(3, hid_nur) / np.sqrt(hid_nur)
        }
        hyper_param = {'learning_rate' : lr,
                        'num_hidden_neuron':hid_nur,
                        'decay_rate': dr,
                        'weights': _w
                        }
        
        _net = Network(hyper_param)
        
        pop.put(_net)


    #prepare the enviroments
    for e in range(NUM_NETWORKS):
        #create a envrioment for each network
        envs.append(gym.make("Pong-v0"))

    #place where we put work that needs to be processed
    workQueue = Queue()
    #thread synchonize lock
    workQueueLock = threading.Lock()
    thExitController = ExitObject()
    #create our threads first
    threads = []
    for threadID in range(NUM_NETWORKS):
        th = WorkThread(threadID, thExitController, workQueueLock, workQueue)
        threads.append(th)
        th.start()




    for generation in range(NUM_GENERATIONS):
        #for each generation
        
        
        print("Starting Generation-%s" % generation)


        result_arr = []
        #starts our threads, gives each thread a network and an enviroment
        workQueueLock.acquire()
        print("Size of ppulation %s" % pop.qsize())
        print("Size of work queue %s" % workQueue.qsize())
        for i in range(NUM_NETWORKS):
            
            #get the lock when putting work into the work queue
            
            work = {
                'net': pop.get(),
                'env': envs[i],
                'result_array': result_arr
                }
            #put the work into the work queue, to let threads process them
        
            workQueue.put(work)
        workQueueLock.release()
        print("pop size: %s " % pop.qsize())
        #put work into the work queue again

        
        
        
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
        file = open("Hyper_param_P1.p","wb")
        print("saving parent 1 and parent 2 weights")
        pickle.dump(parents[0]._hyper_param, file)
        file1 = open("Hyper_param_P2.p","wb")
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

        result_arr.extend(children) # merge existing Networks with chldren
        result_arr.extend(parents)
        for _net in result_arr:
            pop.put(_net)

    #now the remainer of the result_arr has the loswers , we need to mutate some of them and remove
    #however many we are thinking of breeding from the winners

################# we have the two winning parents, kill off 10(50%) and repopulate with spawn






    thExitController.setExit()
    for th in threads:
        th.join()
main()
print("Finished ")




