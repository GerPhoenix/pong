from gym import register

register(
    id="pong-v0",
    entry_point="gym_repo.envs:PongEnv"
)
