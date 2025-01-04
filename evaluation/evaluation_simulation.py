import argparse
import numpy as np
import torch
from util_rgb import make_env
from statistics.experiment_data_collector import ExperimentDataCollector
from util_rgb import make_env, position2road, roads2t_i

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualise the simulation')
    return parser.parse_args()

def initialize_environment(agent_interfaces, scenarios, headless, seed):
    """Initializes the simulation environment."""
    return make_env("smarts.env:hiway-v1", agent_interfaces, scenarios, headless, seed)


def collect_scenario_data(env, agents, central_agent, num_steps, data_collector, index):
    """Runs a single scenario and collects data."""
    print('-'*80)
    print('Starting scenario ', index)
    env.set_scenario(index)
    observations, infos = env.reset()
    terminated = {agent_id: False for agent_id in agents.keys()}
    truncated = {agent_id: False for agent_id in agents.keys()}

    turning_intentions = {"social" : "genral"}

    for agent_id in agents.keys():
        x1 = infos[agent_id]['env_obs'][5].mission.start.position.x
        y1 = infos[agent_id]['env_obs'][5].mission.start.position.y
        start = position2road([x1, y1])

        x2 = infos[agent_id]['env_obs'][5].mission.goal.position.x
        y2 = infos[agent_id]['env_obs'][5].mission.goal.position.y
        goal = position2road([x2, y2])

        turning_intentions[agent_id] = roads2t_i[start + goal]

    data_collector.start_new_scenario(turning_intentions)

    data_collector.mark_agent_succeeded("social")

    for steps in range(num_steps):
        agent_actions = {}
        if central_agent:
            agent_actions = central_agent.act(observations, infos, [agent_id for agent_id in agents.keys() if agent_id in observations and not terminated[agent_id] and not truncated[agent_id]])
        else:
            for agent_id, agent in agents.items():
                if agent_id in observations and not terminated[agent_id] and not truncated[agent_id]:
                    agent_actions[agent_id] = agent.act(observations[agent_id], turning_intentions[agent_id])

        observations, rewards, terminated, truncated, infos = env.step(agent_actions)

        for agent_id in agents.keys():
            if agent_id in observations:
                speed = infos[agent_id]['env_obs'].ego_vehicle_state.speed
                acceleration = infos[agent_id]['env_obs'].ego_vehicle_state.linear_acceleration[0]
                dt = infos[agent_id]['env_obs'].dt
                travel_distance = infos[agent_id]['env_obs'].distance_travelled
                is_waiting = (speed < 0.1)

                data_collector.record_agent_data(
                    agent_id,
                    speed=speed,
                    acceleration=acceleration,
                    dt=dt,
                    travel_distance=travel_distance,
                    is_waiting=is_waiting
                )

                if infos[agent_id]['env_obs'].events.collisions:
                    data_collector.mark_agent_crashed(agent_id)

                if infos[agent_id]['env_obs'].events.reached_goal:
                    data_collector.mark_agent_succeeded(agent_id)
        for social_traffic in infos["social_traffic"]:
             data_collector.record_agent_data(
                    "social",
                    speed=social_traffic["speed"],
                    acceleration=social_traffic["linear_acceleration"][0],
                    dt=social_traffic["dt"],
                    travel_distance=social_traffic["travel_distance"],
                    is_waiting=(social_traffic["speed"] < 0.1)
                )

        if not observations and not infos["social_traffic"]:
            print('No more observations.')
            break
    print('Closing scenario ', index)

    data_collector.close_scenario()

def run_evaluation_simulation(scenarios, agents_spec, id, seed, headless, central_agent = None):
    """Runs the evaluation simulation across specified scenarios and agents."""
    data_collector = ExperimentDataCollector(id)

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent_interfaces = {agent_id: agent_spec.interface for agent_id, agent_spec in agents_spec.items()}
    try:
        agents = {agent_id: agent_spec.build_agent() for agent_id, agent_spec in agents_spec.items()}
    except:
        agents = {agent_id: agent_spec for agent_id, agent_spec in agents_spec.items()}
    env = initialize_environment(agent_interfaces, scenarios, headless, seed)
    num_steps = 2000

    for index, scenario in enumerate(scenarios):
        collect_scenario_data(env, agents, central_agent, num_steps, data_collector, index)


    data_collector.save_raw_data()
    print(f'Finished evaluation with seed {seed}')
    env.close()





