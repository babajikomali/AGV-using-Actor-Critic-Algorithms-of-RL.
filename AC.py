import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gym
import cv2
import pygame 
import matplotlib as plt
from matplotlib import animation

from Differential_Car import Car
from New_Environment import Env, Path
from tqdm import tqdm

REWARDS = []

action_space = {
    0: 'left_increase',
    1: 'left_decrease',
    2: 'right_increase',
    3: 'right_decrease',
    4: 'none'
}

path1 = Path()
env = Env(path1)

#env = gym.make('CartPole-v1')

TOTAL_EPISODES = 30000+1
MAX_STEPS_PER_EPISODE = 10000
LEARNING_RATE = 7e-3

'''
    Actor and Critic Networks with 
    128 units in first hidden layer,
    128units in second hidden layer.
    5 output classes for actor network
    1 output class for ccritic network.
'''
class Critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(256, activation='relu')
    #self.d2 = tf.keras.layers.Dense(128, activation='relu')
    self.v = tf.keras.layers.Dense(1, activation=None)
  def call(self, input_data, training = False):
    x = self.d1(input_data)
    #x = self.d2(x)
    v = self.v(x)
    return v

'''class Critic:
    def __init__(self):
        self.inputs = tf.keras.Input(shape=(4,))
        self.x1 = tf.keras.layers.Dense(2048, activation=tf.nn.relu)(self.inputs)
        self.x2 = tf.keras.layers.Dense(1536, activation=tf.nn.relu)(self.x1)
        self.outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(self.x2)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
    def call(self, input_data):
        self.model(self.input_data)
        return self.model'''

class Actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(256, activation='relu')
    #self.d2 = tf.keras.layers.Dense(128, activation='relu')
    self.a = tf.keras.layers.Dense(5, activation='softmax')
  def call(self, input_data, training = False):
    x = self.d1(input_data)
    #x = self.d2(x)
    a = self.a(x)
    return a


class Agent:
    '''
        Optimizer being used for gradient descent in neural networks is Adam
        Discount Factor is 0.99
    '''
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.c_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.actor = Actor()
        self.critic = Critic()
        self.log_prob = None
    '''
        The tensor received from the actor network is converted to numpy array
        action probabilities which is the output of the neural network are converted 
        as a categorical distribution and a action will be selected at random action 
        will be choosen which is biased
        action tensor then converted into a numpy array and returned  
    '''
    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype = tf.float32)
        action = dist.sample()
        #print(prob)
        #print(dist)
        return int(action.numpy()[0])
    #actor loss is the log probability of action taken multiplied by temporal difference
    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype = tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss
    
    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        '''
            Inputs : present state, action, reward, nextstate, done
            action probabilities are taken from actor network
            present state and next state values are taken from critic network
            one step return is calculated along with the baseline i.e. present state value function
            actor loss is the log probability of action taken multiplied by temporal difference  
            -ve sign is used for calculating bacause we use gradient descent for minimizing loss
            critic loss is the square of the temporal difference because eventually the value 
            of the present state must approach the value of the next state   
            Then the gradients are computed and parameters are updated for both networks
        '''
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            p = self.actor(state, training=True)

            v = self.critic(state, training=True)

            vn = self.critic(next_state, training=True)
            
            td = reward + self.gamma*vn*(1-int(done)) - v
            
            a_loss = self.actor_loss(p, action, td)
            
            c_loss = td**2

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(
            zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

    '''def save(self,t):
        tf.keras.models.save_model(self.actor,f'saved_models/actor_E{t}')
        tf.keras.models.save_model(self.critic,f'saved_models/critic_E{t}')
        self.actor.save(f'saved_models/actor_E{t}.hp5')
        self.critic.save(f'saved_models/critic_E{t}.hp5')'''

agent = Agent()

#os.mkdir('saved_videos')  
#os.mkdir('saved_models')
'''def save_frames_as_gif(frames, path, filename):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)'''

for t in tqdm(range(TOTAL_EPISODES)):
    '''
        start of each episode ---> environment set to reset
        total_reward from the episode set to 0
        Loop through till the end of the episode 
        episode will continue till count reaches the max #steps per episode
    '''
    done = False
    state = env.reset()

    total_reward = 0
    all_aloss = []
    all_closs = []

    frames = []

    count_steps = 0

    while not done :
        '''
            Rendering and Recording of the environment will happen at particular instances
            next state, reward, done will be taken by taking an action following policy network
            actor loss and critic loss for each episode will be stored
            total reward for each episode will be stored
        '''
        '''if not t%1000:
            env.record_env(f'saved/E{t}')'''
        #if not t%100:
        env.render_env(FPS_lock=None)
        action = agent.act(state)
        #print(action,end='|')
        #print(action, end='')
        #next_state, reward, done = env.step(action_space[action],count_steps,MAX_STEPS_PER_EPISODE)
        next_state, reward, done = env.step(action_space[action],count_steps,MAX_STEPS_PER_EPISODE)
        next_state, reward, done, _ = env.step(
            action_space[action])
        aloss, closs = agent.learn(state, action, reward, next_state, done)
        count_steps = count_steps+1
        all_aloss.append(aloss)
        all_closs.append(closs)
        state = next_state
        total_reward += reward
        REWARDS.append(total_reward)
        env.close_quit()
        if done:
            REWARDS.append(total_reward)
            if not t%500:
                print("total reward after {} steps is {}".format(t, total_reward))

REWARDS = np.array(REWARDS)
np.save('saved_models/reward.npy', REWARDS)

