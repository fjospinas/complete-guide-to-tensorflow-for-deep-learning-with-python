# Gotta import gym!
import gym

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('CartPole-v0')

# Reset the environment to default beginning
env.reset()

# Using _ as temp placeholder variable
for _ in range(1000):
    # Render the env
    env.render()

    # Still a lot more explanation to come for this line!
    env.step(env.action_space.sample()) # take a random action
