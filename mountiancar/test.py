from ppo import test
import gymnasium as gym


if __name__ == "__main__":
   
   env = gym.make("Pendulum-v1", render_mode = "human").env

   test(env,"pendulum_train.pth")
