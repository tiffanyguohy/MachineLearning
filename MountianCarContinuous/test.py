from ppo import test
import gymnasium as gym


if __name__ == "__main__":
   
   env = gym.make("MountainCarContinuous-v0", render_mode = "human").env

   test(env,"MountainCar.pth")