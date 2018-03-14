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
import threading

import numpy as np

from helper_methods import preprocess_observation, discount_with_rewards, compute_gradient, \
    update_weights

#BATCH_SIZE = 1 # number of rounds before we update our network

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3

def getAction(action):
    if action == 0:
        return 2 #go up
    elif (action == 1):
        return 0
    else:
        return 3
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
        environment = work_obj['env']
        network_model = work_obj['net']
        obs = environment.reset()
        reward_sum = 0
        prev_processed_obser = None
        running_reward = None
        exp_gradient_squared = {}
        gradient_dict = {}
        gamma = .99

        for layer in network_model.getHyperParam()['weights'].keys():
            # basically populating the expected Gradient values, used for learning later on
            # populate the gradient squared, with all weights in all alyers
            exp_gradient_squared[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])
            gradient_dict[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])

        ep_hidden_layer_vals = []
        ep_obs = []
        ep_gradient_log_ps = []
        ep_rewards = []
        num_rounds = 0
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
                #print("Gradient reward with discount array " , ep_gradient_log_ps_discounted.shape)


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
                    print("updating weights")
                    update_weights(network_model.getHyperParam()['weights'], exp_gradient_squared, gradient_dict,
                               network_model.getHyperParam()['decay_rate'], network_model.getHyperParam()['learning_rate'])

                    print("thread-%s has finished playing... reward is: %s" % (self.threadID, reward_sum))
                    # put the network back into the result array
                    network_model.reward = reward_sum
                    self.QueueLock.acquire()
                    work_obj['result_array'].append(network_model)
                    self.QueueLock.release()

                    #prev_processed_obser = None
                    break

                ep_hidden_layer_vals, ep_obs, ep_gradient_log_ps, ep_rewards = [], [], [], [] # reset the current round values
                obs = environment.reset()
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                reward_sum = 0
                prev_processed_obser = None