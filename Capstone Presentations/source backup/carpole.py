#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   carpole.py
#=#| Date:   3/11/2018
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

import numpy as np
#import cPickle as pickle
import pickle
import gym

import time, threading

# backend
be = 0

# hyperparameters
A = 2  # 2, 3 for no-ops
H = 200  # number of hidden layer neurons
update_freq = 10
batch_size = 1000  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.98  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = 0  # resume from previous checkpoint?
render = 1
device = 1

# model initialization
D = 4  # input dimensionality: 80x80 grid
# if resume:
#     model = pickle.load(open('save.p', 'rb'))
#     print('resuming')
# else:
model = {}
model['W1'] = np.random.randn(D, H) / np.sqrt(D)  # "Xavier" initialization
model['W2'] = np.random.randn(H, A) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in
               model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def softmax(x):
    # if(len(x.shape)==1):
    #  x = x[np.newaxis,...]
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def policy_forward(x):
    #rint(x)
    #print(x.shape)
    if (len(x.shape) == 1):
        x = x[np.newaxis, ...]

    h = x.dot(model['W1'])
    # print("W1 ", model['W1'].shape)
    # print('X ', x.shape)
    # print("h ", h.shape)
    h[h < 0] = 0  # ReLU nonlinearity
    #print("W2 ",  model['W2'].shape)

    logp = h.dot(model['W2'])
    #print("logp ", logp.shape)

    # p = sigmoid(logp)
    p = softmax(logp)

    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = eph.T.dot(epdlogp)
    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0  # backprop relu

    t = time.time()

    dW1 = epx.T.dot(dh)

    #print((time.time() - t0) * 1000, ' ms, @final bprop')

    return {'W1': dW1, 'W2': dW2}


env = gym.make("CartPole-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    t0 = time.time()

    if render:
        t = time.time()
        env.render()
        #print((time.time() - t) * 1000, ' ms, @rendering')

    t = time.time()
    # preprocess the observation, set input to network to be difference image
    x = np.reshape(observation, [1, D])
    print("X shape", x.shape)
    # print((time.time()-t)*1000, ' ms, @prepo')


    # forward the policy network and sample an action from the returned probability
    t = time.time()
    aprob, h = policy_forward(x)
    # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    # print((time.time()-t)*1000, ' ms, @forward')

    # roll the dice, in the softmax loss
    u = np.random.uniform()
    aprob_cum = np.cumsum(aprob)
    a = np.where(u <= aprob_cum)[0][0]
    action = a
    #print("a ", a)
    #print("Action ", action)
    # print(u, a, aprob_cum)


    # record various intermediates (needed later for backprop)
    t = time.time()
    xs.append(x)  # observation
    hs.append(h)  # hidden state

    # softmax loss gradient
    dlogsoftmax = aprob.copy()
    #(1,2)


    #print("logsof ", aprob.shape)
    dlogsoftmax[0, a] -= 1  # -discounted reward
    #print("dlop " ,dlogsoftmax.shape)
    #(1,2) from output a
    dlogps.append(dlogsoftmax)

    # step the environment and get new measurements
    t = time.time()
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    # print((time.time()-t)*1000, ' ms, @env.step')


    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    # print((time.time()-t0)*1000, ' ms, @whole.step')

    if done:  # an episode finished
        episode_number += 1

        t = time.time()

        # stack together all inputs, hidden states, action gradients, and rewards for this episode

        print("length of Obersations " , len(xs))
        print("length of hidden layer values ", len(hs))
        print("length of gradient array ", len(dlogps))
        print("length of rewards array ", len(drs))

        #print("dlogsps " , len(dlogps))
        epx = np.vstack(xs)
        #print("epx " , epx.shape)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        #print("Shape of stacked dlogps " , epdlogp.shape)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory


        print("VSTACK")
        print("VSTACK hidden layer values " , eph.shape)
        print("VSTACK obersation values " , epx.shape)
        print("VSTACK gradient " , epdlogp.shape)
        print("VSTACK of rewards : ", epr.shape)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        #print("eph hidden state" , eph.shape)
        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)

        #print("W1 grad : " , grad['W1'].shape)
        #print("W2 grad : " , grad['W2'].shape)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % update_freq == 0:  # update_freq used to be batch_size
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        #print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        #f episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        observation = env.reset()  # reset env
        prev_x = None

        #print((time.time() - t) * 1000, ' ms, @backprop')

        if episode_number % 10 == 0:  # Pong has either +1 or -1 reward exactly when game ends.
            #            print ('ep %d: game finished, reward: %f' % (episode_number, reward_sum)) + ('' if reward == -1 else ' !!!!!!!!')
            print("HELO")
    reward_sum = 0