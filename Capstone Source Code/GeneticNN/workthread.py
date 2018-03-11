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
from helper_methods import preprocess_observation, discount_with_rewards, compute_gradient, update_weights


class WorkThread(threading.Thread):
    # still needs works
    def __init__(self, tID, exit, lock, workQueue):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.exitFlag = exit
        self.QueueLock = lock
        self.work_queue = workQueue

    def run(self):
        print("Thread- %s, has been started.. waiting for work" % self.threadID)
        while not self.exitFlag.exit:
            self.QueueLock.acquire()
            # print("Thread-%s got the LOCK!" % self.threadID)
            if not self.work_queue.empty():
                work = self.work_queue.get()
                # got the work ,{'net','env','result_array'}
                self.QueueLock.release()
                print("Thread-%s has network: %s" % (self.threadID, work['net'].__str__()))
                self.doWork(work['net'], work['env'], work['result_array'])
            else:
                self.QueueLock.release()

    def doWork(self, network, enviroment, result_array):
        print("Thread-%s doing work..." % self.threadID)
        obs = enviroment.reset()
        reward_sum = 0
        prev_processed_obser = None
        running_reward = None

        exp_gradient_squared = {}
        gradient_dict = {}
        gamma = .99

        for layer in network.getHyperParam()['weights'].keys():
            # basically populating the expected Graident values, used for learning later on
            # populate the gradient squared, with all weights in all alyers
            exp_gradient_squared[layer] = np.zeros_like(network.getHyperParam()['weights'][layer])
            gradient_dict[layer] = np.zeros_like(network.getHyperParam()['weights'][layer])

        ep_hidden_layer_vals = []
        ep_obs = []
        ep_gradient_log_ps = []
        ep_rewards = []

        while True:
            # self.QueueLock.acquire()
            # enviroment.render()
            # self.QueueLock.release()
            # print("Thread-%s WOrking and rendering" % self.threadID)
            processed_obs, prev_processed_obser = preprocess_observation(obs, prev_processed_obser, 80 * 80)

            # hidden_lay_values, action = network.predict(
            #a1 is the output for the hidden layer and a2 is output for the final output layer
            a1, a2 = network.predict(processed_obs)

            ep_obs.append(processed_obs)

            ep_hidden_layer_vals.append(a1)

            action = network.getAction(a2)

            obs, reward, done, info = enviroment.step(action)

            # print("action: %s " % action)
            # print("reward: %s reward sum: %s" % (reward,reward_sum))
            reward_sum += reward

            ep_rewards.append(reward)

            # print("thread-%s reward: %s | done: %s | " %( self.threadID, reward, done))
            if (reward > 0.0):
                print("REWARDED %s" % reward)

            fk_label = np.argmax(a2)  # a2 is a vector, get the index to repersent our label, 0, 1,2
            loss_function_gradient = fk_label - a2[fk_label]
            # print("Loss function gradient : %s "  % loss_function_gradient)

            ep_gradient_log_ps.append(loss_function_gradient)
            if done:
                # episode_numer += 1
                # print("Dont playing around: %s" % reward_sum)

                # form a single entity for all our recorded values for this game
                del ep_hidden_layer_vals[0]
                ep_hidden_layer_vals = np.vstack(ep_hidden_layer_vals)
                del ep_obs[0]
                ep_obs = np.vstack(ep_obs)

                del ep_gradient_log_ps[0]
                ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
                del ep_rewards[0]
                ep_rewards = np.vstack(ep_rewards)

                # change the gradient of the log_ps based on discount rewards

                ep_gradient_log_ps_discounted = discount_with_rewards(ep_gradient_log_ps, ep_rewards, gamma)

                gradient = compute_gradient(ep_gradient_log_ps_discounted,
                                            ep_hidden_layer_vals,
                                            ep_obs,
                                            network.getHyperParam()['weights'])

                # print("gradient size 1", gradient['1'].shape)
                # print("gradient_dict size 1", gradient_dict['1'].shape)
                #               print("gradient size 2 size ", gradient['2'].shape)
                #               print("grandient_dict 2 size",gradient_dict['2'].shape)
                #               for layer in gradient:
                #    gradient_dict[layer] += gradient[layer]

                # gradient_dict['1'] = gradient['1']
                # gradient_dict['2'] = gradient['2']
                # exp_gradient_squared['1'] = gradient['1']
                # exp_gradient_squared['2'] = gradient['2']



                # non batch updates
                update_weights(network.getHyperParam()['weights'], exp_gradient_squared, gradient_dict,
                               network.getHyperParam()['decay_rate'], network.getHyperParam()['learning_rate'])

                print("thread-%s has finished playing... reward is: %s" % (self.threadID, reward_sum))
                # put the network back into the result array
                network.reward = reward_sum
                self.QueueLock.acquire()
                result_array.append(network)
                self.QueueLock.release()

                prev_processed_obser = None
                break

