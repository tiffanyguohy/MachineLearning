import matplotlib.pyplot as plt
import gymnasium as gym
from ppo import train




if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    mean_rewards = train(env,
          save= "MountainCar.pth")
    

    plt.plot( mean_rewards, marker='o', linestyle='-', color='b')
    plt.title("title")
    plt.xlabel("runs")
    plt.ylabel("mean_rewards")
    plt.show()
