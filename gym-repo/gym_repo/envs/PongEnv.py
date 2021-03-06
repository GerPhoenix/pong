import gym
import numpy as np
import pygame
from gym import spaces

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
X_DIMENSION = 800
Y_DIMENSION = 600
SIZE = (X_DIMENSION, Y_DIMENSION)
PLATFORM_VELOCITY = 7
BALL_VELOCITY_X = 28
BALL_VELOCITY_Y = 14

BALL_SIZE = 15
PLATFORM_SIZE_X = 100
PLATFORM_SIZE_Y = 23
PLATFORM_SIZE_Y -= (Y_DIMENSION - PLATFORM_SIZE_Y) % BALL_VELOCITY_Y
PLATFORM_Y_DEFAULT = Y_DIMENSION - PLATFORM_SIZE_Y


class PongEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # ENV INIT
        self.previous_action = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 800, shape=(5,), dtype=np.float32)

        # Init by reset
        self.screen = None
        self.clock = None

        # STATES
        self.score = 0
        # Starting coordinates of the paddle
        self.platform_x = np.random.randint(X_DIMENSION - PLATFORM_SIZE_X)
        self.platform_y = PLATFORM_Y_DEFAULT
        # initial velocity of the paddle
        self.platform_change_x = 0
        self.platform_change_y = 0
        # initial position of the ball
        self.ball_x = np.random.randint(X_DIMENSION)
        self.ball_y = 560
        # velocity of the ball
        # Random direction by x_ball_velocity
        self.ball_change_x = (np.random.randint(0, 2) * 2 - 1) * np.random.randint(BALL_VELOCITY_X - 10,
                                                                                   BALL_VELOCITY_X)
        self.ball_change_y = -BALL_VELOCITY_Y

        # flag for end of episode (reset to start position)
        self.episode_over = False

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        old_score = self.score
        if action == 0:
            self.platform_change_x = 0
        else:
            if action == 1:
                self.platform_change_x = -PLATFORM_VELOCITY
            else:
                self.platform_change_x = PLATFORM_VELOCITY
            self.move_platform()
        self.move_ball()
        reward = self.calculate_reward(old_score, action)
        self.previous_action = action
        # if ball was caught 10 times end the episode
        if self.score == 10:
            self.episode_over = True
        return self.pack_observation(), reward, self.episode_over, {}

    def pack_observation(self):
        return np.array(
            [self.get_platform_x(), self.get_ball_x(), self.ball_change_x, self.ball_y,
             self.ball_change_y])

    def reset(self):
        self.platform_x = np.random.randint(X_DIMENSION - PLATFORM_SIZE_X)
        self.platform_y = PLATFORM_Y_DEFAULT
        # initial velocity of the paddle
        self.platform_change_x = 0
        # initial position of the ball
        self.ball_x = np.random.randint(X_DIMENSION)
        self.ball_y = 560
        # velocity of the ball
        # Random direction by x_ball_velocity
        self.ball_change_x = (np.random.randint(0, 2) * 2 - 1) * np.random.randint(BALL_VELOCITY_X - 10,
                                                                                   BALL_VELOCITY_X)
        self.ball_change_y = -BALL_VELOCITY_Y
        self.score = 0
        # flag for end of episode (reset to start position)
        self.episode_over = False
        return self.pack_observation()

    def render(self, mode='human'):
        if self.screen is None:
            # PYGAME INIT
            pygame.init()
            self.screen = pygame.display.set_mode(SIZE)
            self.clock = pygame.time.Clock()
        self.screen.fill(BLACK)
        self.draw_platform()
        self.draw_ball()
        # score board
        font = pygame.font.SysFont('Calibri', 15, False, False)
        text = font.render("Score = " + str(self.score), True, WHITE)
        self.screen.blit(text, [600, 100])

        pygame.display.flip()
        self.clock.tick(120)

    def close(self):
        pygame.quit()

    # moves the platform. Also restricts its movement between the edges of the window.
    def move_platform(self):
        self.platform_x += self.platform_change_x
        self.platform_y += self.platform_change_y
        if self.platform_x < 0:
            self.platform_x = 0
        if self.platform_x >= X_DIMENSION - PLATFORM_SIZE_X:
            self.platform_x = X_DIMENSION - PLATFORM_SIZE_X - 1

    def draw_platform(self):
        pygame.draw.rect(self.screen, RED, [self.platform_x, self.platform_y, PLATFORM_SIZE_X, PLATFORM_SIZE_Y])

    def draw_ball(self):
        pygame.draw.rect(self.screen, WHITE, [self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE])

    def move_ball(self):
        self.ball_x += self.ball_change_x
        self.ball_y += self.ball_change_y
        if self.ball_x < 0:
            self.ball_x = 0
            self.ball_change_x = self.ball_change_x * -1
        elif self.ball_x > X_DIMENSION - BALL_SIZE:
            self.ball_x = X_DIMENSION - BALL_SIZE
            self.ball_change_x = self.ball_change_x * -1

        if self.ball_y < 0:
            self.ball_y = 0
            self.ball_change_y = self.ball_change_y * -1
        elif -(BALL_SIZE / 2) + self.platform_x < self.get_ball_x() < \
                self.platform_x + PLATFORM_SIZE_X + BALL_SIZE / 2 \
                and PLATFORM_Y_DEFAULT - PLATFORM_SIZE_Y < self.ball_y < PLATFORM_Y_DEFAULT:
            self.ball_y = PLATFORM_Y_DEFAULT - self.ball_change_y
            self.ball_change_x = (self.get_ball_x() - self.get_platform_x()) \
                                 / PLATFORM_SIZE_X * BALL_VELOCITY_X * 2
            self.ball_change_y = self.ball_change_y * -1
            self.score += 1
        elif self.ball_y > PLATFORM_Y_DEFAULT + self.ball_change_y:
            self.episode_over = True

    def get_ball_x(self):
        # Add half the ball_size because the render will be to the right
        return self.ball_x + BALL_SIZE / 2

    def get_platform_x(self):
        # Add half of the platform length because the render will be to the right
        return self.platform_x + PLATFORM_SIZE_X / 2

    def calculate_reward(self, old_score, action):
        reward = 0
        # penalty for doing different actions (smoothing movements)
        if self.previous_action != action:
            reward -= 1
        # if ball was missed apply penalty, if ball was caught apply reward
        if self.episode_over or old_score < self.score:
            reward = 1 / max(abs(self.get_ball_x() - self.get_platform_x()), 1)
        return reward
