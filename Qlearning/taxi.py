import gymnasium as gym
from ql import QLAgent

if __name__ == "__main__":

    print("urmom")
    env = gym.make("Taxi-v3").env
    play_env = gym.make("Taxi-v3", render_mode = "human").env
    agent = QLAgent(env, play_env)

    agent.train(100000, 0.5, 0.9, 0.2)
    print("finished training")

    agent.play()
    agent.play()
    agent.play()
    agent.play()
    agent.play()

