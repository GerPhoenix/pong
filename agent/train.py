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
SAVEFILE_FOLDER = "data/" + "32x32x32new"
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
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
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
cp_callback = ModelCheckpoint(filepath=SAVEFILE_FOLDER + "/dqn_pong_params.h5f",
                              save_weights_only=True,
                              verbose=1)


def training_phase(learn_rate, adam_learn_rate, epsilon, decay, steps):
    policy = DecayingEpsGreedyQPolicy(eps=epsilon, decay=decay)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=learn_rate, policy=policy, gamma=.999)
    dqn.compile(Adam(lr=adam_learn_rate), metrics=['mae'])
    dqn.fit(env, nb_steps=steps, verbose=2, visualize=VISUALIZE, callbacks=[cp_callback])
    dqn.save_weights(SAVEFILE_FOLDER + "/dqn_pong_params.h5f", overwrite=True)


i = 0
# print("Iteration: ", i)
# for _ in range(2):
#     training_phase(0.1, 1e-3, 0.9, 0, 100000)
#     training_phase(0.08, 1e-3, 0.8, 0, 50000)
#     training_phase(0.05, 1e-3, 0.7, 0, 50000)
#     i += 1
#     print("Iteration: ", i)
# for _ in range(2):
#     training_phase(0.08, 1e-3, 0.7, 0, 100000)
#     training_phase(0.05, 1e-3, 0.5, 0, 50000)
#     training_phase(0.03, 1e-3, 0.4, 0, 50000)
#     i += 1
#     print("Iteration: ", i)
for _ in range(2):
    training_phase(0.05, 1e-3, 0.35, 0, 100000)
    training_phase(0.03, 1e-3, 0.25, 0, 50000)
    training_phase(0.02, 1e-3, 0.1, 0, 50000)
    i += 1
    print("Iteration: ", i)
