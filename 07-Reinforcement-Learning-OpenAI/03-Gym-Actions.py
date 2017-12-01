import gym
env = gym.make('CartPole-v0')
# print(env.action_space.)
# #> Discrete(2)
# print(env.observation_space)
# #> Box(4,)
observation = env.reset()

for t in range(1000):

    env.render()

    cart_pos , cart_vel , pole_ang , ang_vel = observation

    # Move Cart Right if Pole is Falling to the Right

    # Angle is measured off straight vertical line
    if pole_ang > 0:
        # Move Right
        action = 1
    else:
        # Move Left
        action = 0

    # Perform Action
    observation , reward, done, info = env.step(action)
