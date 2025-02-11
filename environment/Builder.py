import gymnasium as gym
from smarts.env.configs.hiway_env_configs import ScenarioOrder
from functools import partial

from .Observation import Observation
from .Reward import Reward
from .StackFrames import StackFrames
from .Scenarios import Scenarios
from .ParallelEnvWithScenario import ParallelEnvWithScenario



def make_env(env_name, agent_interfaces, scenario_path, headless, seed, visdom = False) -> gym.Env:
    # Create environment
    env = gym.make(
        env_name,
        scenarios=scenario_path,
        agent_interfaces=agent_interfaces,
        headless=headless,  # If False, enables Envision display.
        seed=seed,
        scenarios_order = ScenarioOrder.sequential
    )
    agent_names = agent_interfaces.keys()

    env = Reward(env=env, agent_names=agent_names)
    env = Observation(shape=(48, 48, 3), env=env, agent_names=agent_names)
    env = StackFrames(env, repeat=3, agent_names=agent_names)
    env = Scenarios(env, agent_names=agent_names, scenario_path=scenario_path)

    return env




def make_env_parallel(env_name, agent_interfaces, scenario_path, headless, seed, num_env=10, visdom = False) -> gym.Env:
    # Create environment
    agent_names = agent_interfaces.keys()
    def env_constructor(sim_name, seed):
        env = gym.make(
            env_name,
            scenarios=scenario_path,
            agent_interfaces=agent_interfaces,
            headless=headless,  # If False, enables Envision display.
            visdom=visdom,  # If True, enables Visdom display.
            seed=seed,
            scenarios_order = ScenarioOrder.sequential,
            sim_name=sim_name
        )

        env = Reward(env=env, agent_names=agent_names)
        env = Observation(shape=(48, 48, 3), env=env, agent_names=agent_names)
        env = StackFrames(env, repeat=3, agent_names=agent_names)
        env = Scenarios(env, agent_names=agent_names, scenario_path=scenario_path)
        return env
    
    # lambdify
    env_constructor_lambdify = lambda sim_name, seed: env_constructor(sim_name, seed)
    
    # A list of env constructors of type `Callable[[int], gym.Env]`
    sim_name = "sim"
    env_constructors = [
        partial(env_constructor_lambdify, sim_name=f"{sim_name}_{ind}") for ind in range(num_env)
    ]

    env = ParallelEnvWithScenario(
        env_constructors=env_constructors,
        seed=seed,
        auto_reset=False,
    )


    return env
