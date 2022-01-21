from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from fedrl.fedrl_env import FedLearning
import yaml
import logging
import os
from datetime import datetime

#from stable_baselines3.common.vec_env import VecNormalize

if __name__ == "__main__":

    yaml_file_name = "./fedrl_MNIST_lenet5.yml"
    yaml_file = open(yaml_file_name)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    rlConfig = parsed_yaml_file["rl_agent"]
    is_train = rlConfig.get('is_train')

    # Instantiate the env
    env = FedLearning()
    env = make_vec_env(lambda: env, n_envs=1)
    #env = VecNormalize（env）

    # select PPO agent
    model = PPO('MlpPolicy', env)

    # train rl or not?
    if is_train:
        # Train PPO agent
        logging.info("Starting training DRL agent.")
        total_timsteps = rlConfig.get('total_timsteps')
        model.learn(total_timesteps=total_timsteps)
        logging.info("Finished training DRL agent.")

        # Save the trained agent
        now = datetime.now()
        dt_string = now.strftime(dt_string=now.strftime("%b-%m-%Y-%H:%M:%S"))
        model_path = "./DRLagent_" + yaml_file_name + "_" + dt_string
        model.save(model_path)
        logging.info("Model saved to %s.", model_path)
    else:
        # load trained model
        model_path = "./DRLagent"
        logging.info("Loading DRL agent from ", model_path)
        model = model.load()
        logging.info("Loaded DRL agent")

        max_steps = rlConfig.get('max_steps')

        obs = env.reset()
        done = False

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            print("Step {}".format(step + 1))
            #env.render(mode='console')
            print("Action: ", action)
            obs, reward, done, info = env.step(action)
            #print('obs=', obs, 'reward=', reward, 'done=', done)
            #env.render(mode='console')
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                logging.info("Goal achieved! ")  #, "reward=", reward)
                #env.render(mode='console')
                break
            if step == max_steps - 1:
                logging.info("Maximum step reached.")
