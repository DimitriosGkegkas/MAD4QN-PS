import gymnasium as gym
from smarts.env.configs.hiway_env_configs import ScenarioOrder
from functools import partial

from environment.Communication import CommunicationWrapper
from environment.Direction import Direction

from .Observation import Observation
from .Reward import Reward
from .StackFrames import StackFrames
from .Scenarios import Scenarios
from .ParallelEnvWithScenario import ParallelEnvWithScenario
from .InfoWrapper import InfoWrapper
from .SocialAgentsWrapper import SocialAgentsWrapper



def make_env(env_name, agent_interfaces, scenario_path, headless, seed, stack_frames = 4, observation_shape=(32, 32, 3), message_dim=8, message_raw_dim=8) -> gym.Env:
    # Create environment
    env = gym.make(
        env_name,
        scenarios=scenario_path,
        agent_interfaces=agent_interfaces,
        headless=headless,  # If False, enables Envision display.
        visdom=False,  # If True, enables Visdom display.
        seed=seed,
        scenarios_order = ScenarioOrder.sequential
    )
    agent_names = agent_interfaces.keys()
    
    env = SocialAgentsWrapper(env, agent_names=agent_names)
    env = InfoWrapper(env, agent_names=agent_names)
    env = Reward(env=env, agent_names=agent_names)
    env = Observation(shape=observation_shape, env=env, agent_names=agent_names)
    env = StackFrames(env, repeat=stack_frames, agent_names=agent_names)
    env = Direction(env, agent_names=agent_names)
    env = CommunicationWrapper(env, agent_names=agent_names, message_dim=message_dim, message_raw_dim=message_raw_dim)
    env = Scenarios(env, agent_names=agent_names, scenario_path=scenario_path)

    return env




def make_env_parallel(env_name, agent_interfaces, scenario_path, headless, seed, num_env=10, stack_frames = 4,  observation_shape=(32, 32, 3), message_dim=8, message_raw_dim=8) -> gym.Env:
    # Create environment
    agent_names = agent_interfaces.keys()
    def env_constructor(sim_name, seed):
        env = gym.make(
            env_name,
            scenarios=scenario_path,
            agent_interfaces=agent_interfaces,
            headless=headless,  # If False, enables Envision display.
            visdom=False,  # If True, enables Visdom display.
            seed=seed,
            scenarios_order = ScenarioOrder.sequential,
            sim_name=sim_name
        )
        
        env = SocialAgentsWrapper(env, agent_names=agent_names)
        env = InfoWrapper(env, agent_names=agent_names)
        env = Reward(env=env, agent_names=agent_names)
        env = Observation(shape=observation_shape, env=env, agent_names=agent_names)
        env = StackFrames(env, repeat=stack_frames, agent_names=agent_names)
        env = Direction(env, agent_names=agent_names)
        env = CommunicationWrapper(env, agent_names=agent_names, message_dim=message_dim, message_raw_dim=message_raw_dim)
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
