import sys

import gym
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ENV_NAME = 'gym_repo:pong-v0'
SAVEFILE_FOLDER = "data/" + "2048x2048x2048Long"

disable_eager_execution()
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
try:
    model.load_weights(SAVEFILE_FOLDER + "/dqn_pong_params.h5f")
except:
    print("No saved weights found")
# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=100000, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Finally, evaluate our algorithm .
dqn.test(env, nb_episodes=10, visualize=True)
sys.exit(0)
