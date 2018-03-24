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
import pickle
import numpy as np
from helper_methods import preprocess_observation, discount_with_rewards, compute_gradient, update_weights, choose_action


class WorkThread(threading.Thread):
    """
    WorkThread object, inherits the parent threading class of python, which allow us to define our own run method
    for our purpose, each thread will hold a environment = pingpong frame, a neural network, and the result array
    each thread will perform training, updating, book keeping of the running mean, and then once the rounds of training have finished
    will return the networks to the result array where the neural networks will be evaluated on performance, and next
    generations is made
    """
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

        print("Thread-%s doing work on... %s" % (self.threadID, work_obj['net'].__str__()))

        #current network session variables
        environment = work_obj['env']
        network_model = work_obj['net']

        #value for the observation of the pixels from the playing field
        obs = environment.reset()
        #current rounds reward
        reward_sum = 0 #check
        #track of our previous frame vs the current frame
        #current frame - previous frame = difference -> values = change in the position of the pinpong ball
        prev_processed_obser = None #check
        #value to keep for tracking the running mean of each round of training
        running_reward = None #chcke
        #holding list for our batchs gradient values, used to compute how much update is going to be performed on the network
        #its the 'error' in backprop
        exp_gradient_squared = {} #check
        gradient_dict = {}  #check
        gamma = .99 #check

        for layer in network_model.getHyperParam()['weights'].keys():
            # basically populating the expected Gradient values, used for learning later on
            # populate the gradient squared, with all weights in all alyers
            exp_gradient_squared[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])
            gradient_dict[layer] = np.zeros_like(network_model.getHyperParam()['weights'][layer])

        #list to keep track of each rounds events
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

            #the most important part, we need to calculate our 'loss' value, however since this is reinforcement learning
            #we do not have the target value that says this current action is CORRECT,  to remedy this we treat this value as
            # if it 'correct' and then minues the actual probablity to reduce its influence
            fake_lbl = 1 if action == 2 else 0
            loss_func_grad = fake_lbl - up_prob
            ep_gradient_log_ps.append(loss_func_grad)
            if (reward > 0.0):
                print("thread - %s REWARDED %s" % (self.threadID, reward))

            if done:
                #done is triggered to True if the round is over, meaning one player reaches 21 points
                #each point is reached if either player misses the pingpong ball
                #increment the round's we just passed
                num_rounds += 1

                #stack our observed values for this ENTRIE session, each session is essentially a frame, and each frame has an action associated with it

                ep_hidden_layer_vals = np.vstack(ep_hidden_layer_vals)
                ep_obs = np.vstack(ep_obs)
                ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
                ep_rewards = np.vstack(ep_rewards)
                # we need a discount factor with our values to make sure the actions at the very beginning are not as heavily influenced
                # as the actions at the very end
                ep_gradient_log_ps_discounted = discount_with_rewards(ep_gradient_log_ps, ep_rewards, gamma)

                #gradient = backPropt(ep_hidden_layer_vals, ep_gradient_log_ps_discounted, network.getHyperParam()['weights'], ep_obs)
                #calculating the gradient will give us how much direction we should move our weights to the correct location
                #using back propogation
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
                    network_model.setReward(reward_sum)

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
                        print("thread-%s finished training with ... %s" %(self.threadID, work_obj['net']))
                        break
                #reset the values for this current round
                ep_hidden_layer_vals = []
                ep_obs = []
                ep_gradient_log_ps = []
                ep_rewards = []
                #reset the pixels on the match round
                obs = environment.reset() # new round reset the environment
                #keep track of the running average of all rounds thus far
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                #some simple current round statistics
                print("Thread-%s Round Finished, reward total: %s, running mean: %s" %(self.threadID, reward_sum, running_reward))
                #reset the running sum for the current round
                reward_sum = 0
                #reset our previous processed value, for a new round
                prev_processed_obser = None