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



            #a1 is the output for the hidden layer and a2 is output for the final output layer
            a2, a1 = network.predict(processed_obs)
            #print("a " , a1.shape)
            #print("b ", a2.shape)

            #reshape to 1X6400 below on bothe
            #ep_obs.append(processed_obs.reshape(processed_obs.shape[0]))
            ep_obs.append(processed_obs)
            ep_hidden_layer_vals.append(a1)

            #ep_hidden_layer_vals.append(a1.reshape(a1.shape[0]))


            u = np.random.uniform()
            probility_cum = np.cumsum(a2)
            a = np.where(u <= probility_cum)[0][0]

            #print("a " , a)
            # a will be either zero, 1 or 2, depending on the random sample


            #action = network.getAction(a2)
            # print("action: %s " % action)
            action = getAction(a)

            obs, reward, done, info = enviroment.step(action)

            reward_sum += reward
            # print("reward: %s reward sum: %s" % (reward,reward_sum))

            copyofsig = a2.copy()

            copyofsig[0,a] -=1 # reward penilize
            ep_gradient_log_ps.append(copyofsig)

            ep_rewards.append(reward)

            # print("thread-%s reward: %s | done: %s | " %( self.threadID, reward, done))
            if (reward > 0.0):
                print("thread - %s REWARDED %s" % (self.threadID, reward))


            #fk_label = np.argmax(a2)  # a2 is a vector, get the index to repersent our label, 0, 1,2
            #loss_function_gradient = fk_label - a2[fk_label]
            #print("loss_function_gradient " , loss_function_gradient[0])
            #ep_gradient_log_ps.append(loss_function_gradient)


            if done:

                # print("DONE ep_hid_layer ", ep_hidden_layer_vals[0].shape)
                # print("DONE ep_obs ", ep_obs[0].shape)
                # print("DONE ep_gradient ", ep_gradient_log_ps[0])
                # print("DONE ep_rewards ", ep_rewards[0])

                ep_hidden_layer_vals = np.vstack(ep_hidden_layer_vals)
                ep_obs = np.vstack(ep_obs)
                ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
                ep_rewards = np.vstack(ep_rewards)


                #change the gradient of the log_ps based on discount rewards
                # print("VSTACKING")
                # print("VSTACK ep_hid_layer: ", ep_hidden_layer_vals.shape)
                # print("VSTACK ep_obs: ", ep_obs.shape)
                # print("VSTACK ep gradient: ", ep_gradient_log_ps.shape)
                # print("VSTACK ep rewards : ", ep_rewards.shape)


                ep_gradient_log_ps_discounted = discount_with_rewards(ep_gradient_log_ps, ep_rewards, gamma)
                #print("Gradient reward with discount array " , ep_gradient_log_ps_discounted.shape)


                gradient = backPropt(ep_hidden_layer_vals, ep_gradient_log_ps_discounted, network.getHyperParam()['weights'], ep_obs)

                # gradient = compute_gradient(ep_gradient_log_ps_discounted,
                #                             ep_hidden_layer_vals,
                #                             ep_obs,
                #                    network.getHyperParam()['weights'])
                #
                #

                #
                # for k in gradient.keys():
                #      print("Gradient ", k, gradient[k].shape)
                for _layer in gradient:
                    gradient_dict[_layer] += gradient[_layer]
                # non batch updates
                update_weights(network.getHyperParam()['weights'], exp_gradient_squared, gradient_dict,
                               network.getHyperParam()['decay_rate'], network.getHyperParam()['learning_rate'])

                print("thread-%s has finished playing... reward is: %s" % (self.threadID, reward_sum))
                # put the network back into the result array
                network.reward = reward_sum
                self.QueueLock.acquire()
                result_array.append(network)
                self.QueueLock.release()

                #prev_processed_obser = None
                break

