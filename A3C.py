import collections
import tensorflow as tf
import numpy as np
import statistics
import tqdm

import pygame
pygame.init()

from tensorflow.python.keras import layers
from typing import Tuple, List, Sequence, Any
from Differential_Car import Car
from Environment import Env, Path

action_space = {
    0: 'left_increase',
    1: 'left_decrease',
    2: 'right_increase',
    3: 'right_decrease',
    4: 'none'
}

'''
    State = { x coordinate, y coordinate, heading angle, cross-track-error}
    End of the episode : Reach Goal point
                         More Cross-track error
                         Multiple spins
                         Cross the boundaries
'''

path1 = Path()
env = Env(path1)

eps = np.finfo(np.float32).eps.item()
'''
    Input state ---> 128 units ---> 5 outputs ------>Policy Network
    Input state ---> 128 units ---> 1 output  ------>Value Network
'''
class ActorCritic(tf.keras.Model):
    def __init__(self,
                 num_actions: int,
                 num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        #inputs = tf.keras.Input(shape=(4,))
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

num_actions = 5
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 
    state, reward, done = env.step(action_space[action])
    #print(action_space[action])
    #env.render_env(None)
    return (state.astype(np.float32),
            np.array(reward,np.int32),
            np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                                        [tf.float32, tf.int32, tf.int32])

def run_episode(initial_state = tf.Tensor,
                model = tf.keras.Model,
                max_steps = int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)  

    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):

        state = tf.expand_dims(state, 0)     
        action_logits_t, value = model(state)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        values = values.write(t, tf.squeeze(value))

        action_probs = action_probs.write(t, action_probs_t[0, action])
        #print(action_probs)

        state, reward, done = tf_env_step(action)
        
        #env.render_env(None)

        state.set_shape(initial_state_shape)

        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break
        #env.close_quit()

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    
    return action_probs, values, rewards

def get_expected_return(rewards : tf.Tensor,
                        gamma : float,
                        standardize: bool = True) -> tf.Tensor:
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape

    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma*discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i,discounted_sum)
    returns = returns.stack()[::-1]
    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs : tf.Tensor,
                 values : tf.Tensor,
                 returns : tf.Tensor) -> tf.Tensor:
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs*advantage)

    critic_loss = huber_loss(values, returns)
    
    return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

@tf.function
def train_step(initial_state : tf.Tensor,
               model : tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               gamma : float,
               max_steps_per_episode : int) -> tf.Tensor:
    with tf.GradientTape() as tape:

        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode)

        returns = get_expected_return(rewards, gamma)

        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        loss = compute_loss(action_probs, values, returns)
    print('Hi')
    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


#%% time

min_episodes_criterion = 300
max_episodes = 30000
max_steps_per_episode = 1000

reward_threshold = 1000
running_reward = 0

gamma = 0.99

episodes_reward: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
    for i in t:
        print('Hello')
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(train_step(
            initial_state, model, optimizer, gamma, max_steps_per_episode))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        if i % 10 == 0:
            pass #print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


