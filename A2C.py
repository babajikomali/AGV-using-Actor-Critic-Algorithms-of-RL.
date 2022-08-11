import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gym
import cv2
import pygame
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow.python.keras.losses as kls

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
MAX_STEPS_PER_EPISODE = 20000
LEARNING_RATE = 7e-3

class Critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(256, activation='relu')
    self.d2 = tf.keras.layers.Dense(128, activation='relu')
    self.v = tf.keras.layers.Dense(1, activation=None)
  def call(self, input_data, training = False):
    x = self.d1(input_data)
    x = self.d2(x)
    v = self.v(x)
    return v


class Actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(256, activation='relu')
    self.d2 = tf.keras.layers.Dense(128, activation='relu')
    self.a = tf.keras.layers.Dense(5, activation='softmax')

  def call(self, input_data, training=False):
    x = self.d1(input_data)
    x = self.d2(x)
    a = self.a(x)
    return a


class Agent:
    
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.RMSprop(LEARNING_RATE)
        self.c_opt = tf.keras.optimizers.RMSprop(LEARNING_RATE)
        self.actor = Actor()
        self.critic = Critic()
        self.log_prob = None
    

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        print(prob)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):
        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
          dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
          log_prob = dist.log_prob(a)
          prob = dist.prob(a)
          probability.append(prob)
          log_probability.append(log_prob)
        p_loss = []
        e_loss = []
        td = td.numpy()
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

agent = Agent()

#os.mkdir('saved_videos')
#os.mkdir('saved_models')

def preprocess1(states, actions, rewards, gamma):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)

    return states, actions, discnt_rewards


ep_reward = []
total_avgr = []

for t in tqdm(range(TOTAL_EPISODES)):
    
    done = False
    state = env.reset()

    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []

    count_steps = 0

    while not done:

        action = agent.act(state)
        next_state, reward, done = env.step(
            action_space[action], count_steps, MAX_STEPS_PER_EPISODE)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        count_steps = count_steps+1
        state = next_state
        total_reward += reward
        REWARDS.append(total_reward)
        env.close_quit()
        if done:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-500:])
            total_avgr.append(avg_reward)
            states, actions, discnt_rewards = preprocess1(states, actions, rewards, 1)  
            al,cl = agent.learn(states, actions, discnt_rewards)    
            if not t % 500:
                print("total reward after {} steps is {}".format(t, total_reward))
                print(f"al{al}")
                print(f"cl{cl}")

ep = [i  for i in range(TOTAL_EPISODES)]
plt.plot(ep,total_avgr,'b')
plt.title("avg reward Vs episods")
plt.xlabel("episods")
plt.ylabel("average reward per 500 episods")
plt.grid(True)
plt.show()
