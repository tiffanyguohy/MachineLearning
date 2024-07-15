from dql import DQLAgent
import gymnasium as gym



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    play_env = gym.make("CartPole-v1", render_mode = "human").env

    agent = DQLAgent(env, play_env)
    agent.train()
    agent.play()



