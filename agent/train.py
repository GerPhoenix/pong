import gym
import numpy as np
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

from agent.policies.decaying_epsilon_q_greedy import DecayingEpsGreedyQPolicy

ENV_NAME = 'gym_repo:pong-v0'

disable_eager_execution()
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
try:
    model.load_weights("data/dqn_pong_params.h5f")
except:
    print("No saved weights found")
# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
policy = DecayingEpsGreedyQPolicy(eps=0.35)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=0.05, policy=policy)
dqn.compile(Adam(lr=1e-2), metrics=['mae'])

cp_callback = ModelCheckpoint(filepath="data/dqn_pong_params.h5f",
                              save_weights_only=True,
                              verbose=1)
dqn.fit(env, nb_steps=15000, verbose=2, visualize=False, callbacks=[cp_callback])

policy = DecayingEpsGreedyQPolicy(eps=0.2)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=0.035, policy=policy)
dqn.compile(Adam(lr=1e-2), metrics=['mae'])
dqn.fit(env, nb_steps=30000, verbose=2, visualize=False, callbacks=[cp_callback])

policy = DecayingEpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=0.02, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something using the fit function! You can visualize the training here, but this
# slows down training quite a lot.
# You can abort the training without much progress being lost if the cp_callback is bound
#
dqn.fit(env, nb_steps=30000, verbose=2, visualize=False, callbacks=[cp_callback])

policy = DecayingEpsGreedyQPolicy(eps=0.05)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=25000, verbose=2, visualize=False, callbacks=[cp_callback])

# After training is done, we save the best weights.
dqn.save_weights("data/dqn_pong_params.h5f", overwrite=True)
