import pathlib
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from evaluation.evaluation_simulation import run_evaluation_simulation, parse_arguments
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch
from test.test import Vehicle, IntersectionManager


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    class CentralAgent(Agent):
        def __init__(self):
            """Any policy initialization matters, including loading of model,
            may be performed here.
            """
            self.manager = IntersectionManager(intersection_width=14.21, intersection_height=14.21, center=(100, 100))

        def getAction(self, speed):
            return (speed, 0)
        def act(self, observations, infos, agent_names):
            vehicles = []
            agent_actions = {}
            for q, j in enumerate(agent_names):
                if j in list(observations.keys()):
                    vehicles.append(
                        Vehicle(
                            vehicle_id=j, 
                            position=(infos[j]['env_obs'].ego_vehicle_state.position[0], infos[j]['env_obs'].ego_vehicle_state.position[1]), 
                            velocity=infos[j]['env_obs'].ego_vehicle_state.speed, 
                            goal=(infos[j]['env_obs'].ego_vehicle_state.mission.goal.position.x, infos[j]['env_obs'].ego_vehicle_state.mission.goal.position.y)))
            self.manager.add_vehicles(vehicles)
            target_velocities = self.manager.calculate_target_velocities()
            for vehicle_id, velocity in target_velocities.items():
                agent_actions[vehicle_id] = self.getAction(velocity)
            return agent_actions


    # AgentSpec specifying the agent's interface and policy.
    agent_spec = AgentSpec(
        # Agent's interface.
        interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=None, top_down_rgb=True),
        # Agent's policy.
    )

    agents_spec = {
            f"Agent-{i}": agent_spec for i in range(4)  # Supports four agents by default
    }
    
    scenarios_path = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "multi_scenario"
    scenarios = [str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()]
    scenarios.sort()

    central_agent = CentralAgent()


    run_evaluation_simulation(scenarios, agents_spec, "fcfs", args.seed, False, central_agent)



