from dql import DQLAgent
import gymnasium as gym



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQLAgent(env)
    agent.train()



