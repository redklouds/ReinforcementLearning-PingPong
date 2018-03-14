#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   workthread.py
#=#| Date:   3/10/2018
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import threading,pickle
import numpy as np
from helper_methods import preprocess_observation, discount_with_rewards, compute_gradient, \
    update_weights, choose_action

#BATCH_SIZE = 1 # number of rounds before we update our network

def backPropt(hidden_val_stack, gradient_stack, weights, observation_stack):
    dW2 = hidden_val_stack.T.dot(gradient_stack)
    dh = gradient_stack.dot(weights['2'].T)
    dh[hidden_val_stack <= 0] = 0 #derivative of ReLu

    dW1 =  observation_stack.T.dot(dh)

    return {'1':dW1, '2':dW2}


class WorkThread(threading.Thread):
    # still needs works
    def __init__(self, tID, exit, lock, workQueue):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.exitFlag = exit
        self.QueueLock = lock
        self.work_queue = workQueue

    def run(self):
        """
        WorkThread starting point, will start and run a while loop, checking the workQueue for work,
        important, WE NEED a lock  to synchronize the threads prior to checking the queue, once the lock object has
        signaed to stop checking , our threads exit the run method.
        :return:
        """
        print("Thread-%s, has been started.. waiting for work" % self.threadID)
        while not self.exitFlag.exit: # our exit object, needs reference to all threads to hold a boolean value
            # defaulting to False
            self.QueueLock.acquire() # try to get the queue lock
            #this thread has gotten the lock, entering the critical section
            if not self.work_queue.empty():
                #if the work queue is NOT EMPTY
                #get work from the workQueue
                work = self.work_queue.get()

                #work Queue contains {'env','network','num_rounds', result_array}
                self.QueueLock.release()
                #once the work has been taken from the queue release the lock and perform the work
                #self.doWork(work['net'], work['env'], work['result_array'])
                self.doWork(work)
            else:
                self.QueueLock.release()
    def doWork(self, work_obj):
    #def doWork(self, network, enviroment, result_array):
        print("Thread-%s doing work on... %s" % (self.threadID, work_obj['net'].__str__()))


        #current network session variables
        environment = work_obj['env']
        network_model = work_obj['net']
        obs = environment.reset()
        reward_sum = 0 #check
        prev_processed_obser = None #check
        running_reward = None #chcke
        exp_gradient_squared = {} #check
        gradient_dict = {}  #check
        gamma = .99 #check

        for layer in network_model.getHyperParam()['weights'].keys():
            # basically populating the expected Gradient values, used for learning later on
            # populate the gradient squared, with all weights in all alyers
            exp_gradient_squared[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])
            gradient_dict[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])

        ep_hidden_layer_vals = []
        ep_obs = []
        ep_gradient_log_ps = []
        ep_rewards = []
        num_rounds = 0 #check
        while True:

            # self.QueueLock.acquire()
            # enviroment.render()
            # self.QueueLock.release()
            processed_obs, prev_processed_obser = preprocess_observation(obs, prev_processed_obser, 80 * 80)


            #forward feed
            hidd_lay, up_prob = network_model.predict1(processed_obs)

            ep_obs.append(processed_obs)
            ep_hidden_layer_vals.append(hidd_lay)

            #get the action to move the paddle up or otherwise down
            action = choose_action(up_prob)

            obs, reward, done, info = environment.step(action)

            reward_sum += reward
            ep_rewards.append(reward)


            fake_lbl = 1 if action == 2 else 0
            loss_func_grad = fake_lbl - up_prob
            ep_gradient_log_ps.append(loss_func_grad)
            if (reward > 0.0):
                print("thread - %s REWARDED %s" % (self.threadID, reward))

            if done:
                num_rounds += 1
                ep_hidden_layer_vals = np.vstack(ep_hidden_layer_vals)
                ep_obs = np.vstack(ep_obs)
                ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
                ep_rewards = np.vstack(ep_rewards)

                ep_gradient_log_ps_discounted = discount_with_rewards(ep_gradient_log_ps, ep_rewards, gamma)

                #gradient = backPropt(ep_hidden_layer_vals, ep_gradient_log_ps_discounted, network.getHyperParam()['weights'], ep_obs)
                gradient = compute_gradient(
                    ep_gradient_log_ps_discounted,
                    ep_hidden_layer_vals,
                    ep_obs,
                    network_model.getHyperParam()['weights']
                )
                for _layer in gradient:
                    gradient_dict[_layer] += gradient[_layer]
                # non batch updates

                if num_rounds % work_obj['num_rounds'] == 0:
                    #every num_rounds update the weight
                    update_weights(network_model.getHyperParam()['weights'], exp_gradient_squared, gradient_dict,
                               network_model.getHyperParam()['decay_rate'], network_model.getHyperParam()['learning_rate'])

                    print("thread-%s has finished playing... reward is: %s" % (self.threadID, reward_sum))
                    # put the network back into the result array
                    network_model.reward = reward_sum

                    if "tag" in work_obj.keys():
                        print("[++++} Comparison network has finished the batch training , and just updated weights")
                        print("[++++] Stats: Thread-%s running reward: %s" % (self.threadID, reward_sum))
                        #"this is the tagged network"
                        #save its current progress and continue the loop
                        f = open("comparison.p","wb")
                        pickle.dump(network_model.getHyperParam(), f)
                        f.close()

                    else:

                        self.QueueLock.acquire()
                        work_obj['result_array'].append(network_model)
                        self.QueueLock.release()

                        #prev_processed_obser = None

                        break

                ep_hidden_layer_vals, ep_obs, ep_gradient_log_ps, ep_rewards = [], [], [], [] # reset the current round values
                obs = environment.reset() # new round reset the environment
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                #some simple current round statistics
                print("Thread-%s Round Finished, reward total: %s, running mean: %s" %(self.threadID, reward_sum, running_reward))
                reward_sum = 0
                prev_processed_obser = None