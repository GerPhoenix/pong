import gym
import numpy as np
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Activation

ENV_NAME = 'gym_repo:pong-v0'
SAVEFILE_FOLDER = "data/" + "4096x4096x4096"
VISUALIZE = False

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
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(4096))
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
memory = SequentialMemory(limit=1000000, window_length=1)
cp_callback = ModelCheckpoint(filepath=SAVEFILE_FOLDER + "/dqn_pong_params.h5f",
                              save_weights_only=True,
                              verbose=1)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)


def train(learn_rate, model_update_interval, steps):
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
                   target_model_update=model_update_interval, policy=policy, gamma=.99, train_interval=4)
    dqn.compile(Adam(lr=learn_rate), metrics=['mae'])
    dqn.fit(env, nb_steps=steps, verbose=2, visualize=VISUALIZE)
    dqn.save_weights(SAVEFILE_FOLDER + "/dqn_pong_params.h5f", overwrite=True)


train(1e-3, 1e-3, 1750000)
