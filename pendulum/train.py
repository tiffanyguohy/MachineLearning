import matplotlib.pyplot as plt
import gymnasium as gym
from ppo import train




if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    mean_rewards = train(env,
          save= "pendulum_train.pth")
    

    plt.plot( mean_rewards, marker='o', linestyle='-', color='b')
    plt.title("title")
    plt.xlabel("season")
    plt.ylabel("mean_rewards")
    plt.show()
