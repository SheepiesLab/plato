# Reinforcement_Learning: A Reinforcement Learning Framework for Active Federated Learning

_The programs of the framework is located at `plato/utils/reinforcement_learning/`._

## Description to primary files
- `base_rl_agent.py`: A base class for RL agent, adapted from `plato.clients.base`; it communicates with the FL server via socket.io
- `simple_rl_agent.py`: A simple RL agent inherited from the base class `base_rl_agent.py` that uses Gym interface to realize the RL control
- `simple_rl_server.py`: It's a simple FL server inherited from `plato.servers.base` and `plato.servers.fedavg` that conducts the FL training with the control of an RL agent via socket.io

## How to use this framework and customize your own DRL agent for learning and controlling FL?

There is an example `examples/fei` that implements an DRL agent to learn and control the global aggregation step in FL.

Just like it, we can customize another DRL agent by
0. Create a new directory `examples\your_algorithm` and then fulfill the following files in this directory
1. Implement a subclass of `RLAgent` in `simple_rl_agent.py` and its abstract methods `update_policy()` and `process_experience()`; assign your RL policy to `self.policy` in the class initialization; then override `prep_action()`, `get_state()`, `get_reward()`, `get_done()` or other methods for further customization (see `examples/fei/fei_agent.py`)
2. Implement a subclass of `RLServer` in `simple_rl_server.py` and its abstract methods `prep_state()` and `apply_action()`; then override `prep_state()` or other methods for further customization (see `examples/fei/fei_server.py`)
3. If you want to customize other components of FL, it will be the same as what you're supposed to do with plato; e.g., a customized client (see `examples/fei/fei_client.py`) or a customized trainer (see `examples/fei/fei_trainer.py`)
4. Implement a starting program that runs the DRL agent as a subprocess along with the FL processes (see `examples/fei/fei.py` as a template and replace the components as you need)
5. Implement the `xxx.yml` configure file and change the `os.environ['config_file']` in `config.py` to the path of this `xxx.yml` file; you need to pass your `


## How to run your customized DRL agent for learning and controlling FL?

To start the training/testing of your customized DRL-controlled FL training, just simply run the starting program, e.g.,

```
python examples/fei/fei.py 
```

- `config.py`: It fulfills the configuration for the RL agent based on the FL configuration `plato.config.Config`
- `fei_FMNIST_lenet5.yml`: It's an instance of configuration file for running `examples/fei`, which has an RL-based aggregation strategy
- `policies`: under this directory, several RL control policies and NNs have been implemented for experiments

Tune `reinforcement_learning/xxx.yml` file for configuring the FL training sessions, and tune `reinforcement_learning/config.py` file for configuring the policy and training/testing of the RL agent.
