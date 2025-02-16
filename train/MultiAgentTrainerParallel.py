import pathlib
from typing import Union
import numpy as np
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch
from statistics.experiment_data_collector import ExperimentDataCollector
from environment import make_env, make_env_parallel
from utils import position2road, roads2t_i
from datetime import datetime
import sys
import os

class MultiAgentTrainerParallel:
    def __init__(
        self,
        args,
        batch_size=256,
        best_score=-1000.0,
        total_steps=int(1e6),
        agent_count=4,
        algorithm_identifier='DuelingDDQNAgents',
        evaluation_step=10,
        num_env = 1,
        evaluation = False,
    ):
        self.args = args
        self.batch_size = batch_size
        self.best_score = best_score
        self.total_steps = total_steps
        self.agent_count = agent_count
        self.start_time = datetime.now()
        self.evaluation_step = evaluation_step
        self.n_steps = 0
        self.n_episodes = 0
        self.scores_list = []
        self.scores_per_scenario_list = []
        self.num_env = num_env
        
        self.algorithm_identifier = algorithm_identifier
        self.evaluate = evaluation
        self.timestamp = datetime.now().strftime("%d%m%Y")
        self.agents = {}
        if not self.evaluate:
            self.training_stats_path = os.path.join(
                    "training_stats",
                    self.algorithm_identifier,
                    self.timestamp,
            )
            os.makedirs(self.training_stats_path, exist_ok=True)

    def initialize_environment(self, agent_spec, scenario_subdir="scenarios/sumo/multi_scenario", parallel=True):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.agent_names = [f"Agent-{i}" for i in range(self.agent_count)]
        agent_interfaces = {agent_id: agent_spec.interface for agent_id in self.agent_names}

        scenarios_path = pathlib.Path(__file__).absolute().parent.parent / scenario_subdir
        self.scenarios = [str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()]
        self.scenarios.sort()

        if parallel:
            self.env = make_env_parallel("smarts.env:hiway-v1", agent_interfaces, self.scenarios, True, self.args.seed, num_env= self.num_env)
        
        self.single_env = make_env("smarts.env:hiway-v1", agent_interfaces, self.scenarios, False, self.args.seed, True)

    def initialize_agents(
        self,
        batch_size=256,
        gamma=0.99,
        epsilon=1.0,
        lr=0.0001,
        eps_min=0.01,
        replace=1000,
        eps_dec=1e-6,
        mem_size_factor=1.5,
        n_actions=2,
        base_dir='models',
        preload = False
    ):
        mem_size = 1 if self.evaluate else 1e5
        if self.evaluate or preload:
            chkpt_dir = base_dir
            assert os.path.exists(chkpt_dir), f"Checkpoint directory {chkpt_dir} does not exist"
        else:
            chkpt_dir = os.path.join(
                base_dir, self.algorithm_identifier, self.timestamp
            )
            os.makedirs(chkpt_dir, exist_ok=True)

        input_dims = self.env.observation_space.shape
        agent_params = {
            'gamma': gamma,
            'epsilon': epsilon,
            'lr': lr,
            'input_dims': input_dims,
            'n_actions': n_actions,
            'eps_min': eps_min,
            'batch_size': batch_size,
            'replace': replace,
            'eps_dec': eps_dec,
            'chkpt_dir': chkpt_dir,
            'algo': self.algorithm_identifier,
            'mem_size': int(mem_size * mem_size_factor),
            'training_stats_path': self.training_stats_path,
        }
        self.agents = {
            'straight': DuelingDDQNAgent(
                **agent_params,
                env_name=f'agent_straight_{self.args.seed}'
            ),
            'left': DuelingDDQNAgent(
                **agent_params,
                env_name=f'agent_left_{self.args.seed}'
            ),
            'right': DuelingDDQNAgent(
                **agent_params,
                env_name=f'agent_right_{self.args.seed}'
            )
        }

        if self.evaluate or preload:
            self.load_models()

        if preload and not self.evaluate:
            self._set_best_score()

    def load_models(self):
        for agent in self.agents.values():
            agent.load_models()

    def train(self):
        while self.n_steps < self.total_steps:
            self._run_episode()

    def _run_episode(self):
        batch_turning_intentions, batch_observations, batch_terminated, batch_truncated, batch_rewards, batch_infos = self._batch_initialize_episode()
        ep_steps = 0
        batch_score = [0 for _ in range(len(batch_observations))]
        while not self._batch_is_episode_ended(batch_observations, batch_rewards, batch_terminated, batch_infos) and  ep_steps < 1000:
            batch_agent_actions = self._batch_select_actions(batch_turning_intentions, batch_observations, batch_terminated, batch_truncated)
            batch_observations_, batch_rewards, batch_terminated, batch_truncated, batch_infos = self.env.step(
                [ 
                   {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()} for agent_actions in batch_agent_actions
                ]
            )
            batch_score = [sum(rewards) + score for rewards, score in zip(batch_rewards, batch_score)]
            batch_agent_rewards = []
            for observations_, rewards in zip(batch_observations_, batch_rewards):
                filtered_agents = [agent_name for agent_name in self.agent_names if agent_name in observations_]
                agent_rewards = {agent_name: rewards[idx] for idx, agent_name in enumerate(filtered_agents)}
                batch_agent_rewards.append(agent_rewards)


            self._batch_store_transitions(batch_observations, batch_agent_actions, batch_agent_rewards, batch_observations_, batch_terminated, batch_truncated, batch_turning_intentions)

            self._update_agents()
            batch_observations = batch_observations_
            self.n_steps += 1
            ep_steps += 1
            self._log_progress(np.average(batch_score), ep_steps)
        self.n_episodes += 1
        self._evaluate_if_needed()
        
    def _get_turning_intention(self, infos):
        turning_intentions = {}
        for k in self.agent_names:
            start = position2road([infos[k]['env_obs'][5].mission.start.position.x, infos[k]['env_obs'][5].mission.start.position.y])
            goal = position2road([infos[k]['env_obs'][5].mission.goal.position.x, infos[k]['env_obs'][5].mission.goal.position.y])
            turning_intentions[k] = roads2t_i[start + goal]
        return turning_intentions
    
    def _batch_get_turning_intentions(self, batch_infos):
        batch_turning_intentions = []
        for infos in batch_infos:
            turning_intentions = self._get_turning_intention(infos)
            batch_turning_intentions.append(turning_intentions)
        return batch_turning_intentions
        

    def _initialize_episode(self, id = None):
        if id is not None:
            self.single_env.set_scenario(id)
        observations, infos = self.single_env.reset()
        turning_intentions = self._get_turning_intention(infos)
        terminated, truncated = {agent_id: False for agent_id in self.agent_names}, {agent_id: False for agent_id in self.agent_names}
        rewards = [0 for _ in range(len(self.agent_names))]
        return turning_intentions, observations, terminated, truncated, rewards, infos

    def _batch_initialize_episode(self, ids = None):
        if ids is not None:
            self.env.set_scenario(ids)
        batch_observations, batch_infos = self.env.reset()
        batch_turning_intentions = self._batch_get_turning_intentions(batch_infos)
        terminated, truncated = {agent_id: False for agent_id in self.agent_names}, {agent_id: False for agent_id in self.agent_names}
        terminated["__all__"] = False
        truncated["__all__"] = False
        rewards = [0 for _ in range(len(self.agent_names))]
        batch_terminated = [terminated for _ in range(len(batch_observations))]
        batch_truncated = [truncated for _ in range(len(batch_observations))]
        batch_rewards = [rewards for _ in range(len(batch_observations))]
        return batch_turning_intentions, batch_observations, batch_terminated, batch_truncated, batch_rewards, batch_infos


    def _is_episode_ended(self, observations, rewards, terminated, info):
        return  (
                    -10 in rewards # someone crashed
                    or len(observations) == 0 
                    # or all(reward == -1 for reward in rewards) They have to learn not to stop
                    or ("__all__" in terminated and terminated["__all__"])
                    # or ep_steps > 1000
                    
                ) \
                and (not info["social_traffic"])

    
    def _batch_is_episode_ended(self, batch_observations, batch_rewards, batch_terminated, batch_infos):
        """
            Returns true if all the observations foreach batch are empty or the ep_steps is greater than 1000
            also if the rewards are -1 for all the agents for all the scenarios
        """
        return  all([
            self._is_episode_ended(observations, rewards, terminated, info) \
                for observations, rewards, terminated, info \
                    in zip(batch_observations, batch_rewards, batch_terminated, batch_infos)])
    
    def test(self, batch_observations, batch_rewards, batch_terminated, batch_infos, ep_steps):
        """
            Returns true if all the observations foreach batch are empty or the ep_steps is greater than 1000
            also if the rewards are -1 for all the agents for all the scenarios
        """
        return  [
            self._is_episode_ended(observations, rewards, terminated, info) \
                for observations, rewards, terminated, info \
                    in zip(batch_observations, batch_rewards, batch_terminated, batch_infos)]
    
    def act(self, obs, turning_intention):
        return self.agents[turning_intention].choose_action(obs, self.evaluate)
        
    def format_action(self, action):
        return action

    def _batch_select_actions(self, batch_turning_intentions, batch_observations, batch_terminated, batch_truncated):
        batch_agent_actions = []
        for observations, terminated, truncated, turning_intentions in zip(batch_observations, batch_terminated, batch_truncated, batch_turning_intentions):
            agent_action = self._select_actions(turning_intentions, observations, terminated, truncated)
            batch_agent_actions.append(agent_action)
        return batch_agent_actions 
    
    def _select_actions(self, turning_intentions, observations, terminated, truncated):
        agent_actions = {}
        for idx, agent_name in enumerate(self.agent_names):
            if agent_name in observations and not terminated[agent_name] and not truncated[agent_name]:
                agent_actions[agent_name] = self.act(observations[agent_name], turning_intentions[agent_name])
        return agent_actions
    
    def  _batch_store_transitions(self, batch_observations, batch_agent_actions, batch_agent_rewards, batch_observations_, batch_terminated, batch_truncated, batch_turning_intentions):
        for observations, agent_actions, agent_rewards, observations_, terminated, truncated, turning_intentions in zip(batch_observations, batch_agent_actions, batch_agent_rewards, batch_observations_, batch_terminated, batch_truncated, batch_turning_intentions):
            self._store_transitions(observations, agent_actions, agent_rewards, observations_, terminated, truncated, turning_intentions)

    def _store_transitions(self, observations, agent_actions, agent_rewards, observations_, terminated, truncated, turning_intentions):
        for idx, agent_name in enumerate(self.agent_names):
            if agent_name in observations and agent_name in observations_:
                self.agents[turning_intentions[agent_name]].store_transition(observations[agent_name], agent_actions[agent_name], agent_rewards[agent_name], observations_[agent_name], done=terminated[agent_name])

    def _update_agents(self):
        # if self.n_steps % (self.batch_size // 2) == 0:
        for agent in self.agents.values():
            agent.learn()

    def _set_best_score(self):
        self.load_scores()
        scores, scores_per_scenario = self.eval()
        elapsed_time = datetime.now() - self.start_time 
        self.scores_list.append((scores, str(elapsed_time)))
        self.scores_per_scenario_list.append(scores_per_scenario)
        self.best_score = scores
        self.save_scores()

    def _evaluate_if_needed(self):
        if self.n_episodes % self.evaluation_step == 0:
            scores, scores_per_scenario = self.eval()
            elapsed_time = datetime.now() - self.start_time 
            self.scores_list.append((scores, str(elapsed_time)))
            self.scores_per_scenario_list.append(scores_per_scenario)
            if scores > self.best_score:
                for agent in self.agents.values():
                    agent.save_models()
                self.best_score = scores
            self.save_scores()

    def save_scores(self):
        np.save(os.path.join(self.training_stats_path, "avg_reward.npy"), np.array(self.scores_list, dtype=object))
        np.save(os.path.join(self.training_stats_path, "avg_reward_per_scenario.npy"), np.array(self.scores_per_scenario_list))

    def load_scores(self):
        try:
            self.scores_list = np.load(os.path.join(self.training_stats_path, "avg_reward.npy"), allow_pickle=True).tolist()
            self.scores_per_scenario_list = np.load(os.path.join(self.training_stats_path, "avg_reward_per_scenario.npy"), allow_pickle=True).tolist()
        except:
            self.scores_list = []
            self.scores_per_scenario_list = []

    def _log_progress(self, score, ep_steps):
        elapsed_time = datetime.now() - self.start_time
        total_seconds = int(elapsed_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        sys.stdout.write(
            f"\r Batch Episode: {self.n_episodes} | Steps: {ep_steps} | Avg Reward: {score:.2f} | Elapsed Time: {hours:02}:{minutes:02}:{seconds:02}"
        )
        sys.stdout.flush()

    def _log_percentage(self, percentage):
        """
            percentage: float

        """
        dotes_to_display = int(percentage * 20) * "-" + int((1 - percentage) * 20) * "_"
        sys.stdout.write(
            f"\r [{dotes_to_display}] {percentage * 100:.2f}%"
        )
        sys.stdout.flush()
    
    def slice_list(self, lst, n):
        """
        Yields slices of the list, each containing up to n elements.

        Args:
            lst (list): The list to be sliced.
            n (int): The size of each chunk.

        Yields:
            list: A slice of the list with up to n elements.
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def eval(self):
        print("\n----------------------------------------------------------------------------")
        print(f"Starting evaluation phase")
        eval_episodes = len(self.scenarios)
        self.evaluate = True
        batch_rewards = []
        self._log_percentage(0)
        for ids in self.slice_list(list(range(eval_episodes)), self.num_env):
            rewards = self._eval_episodes(ids)
            for reward in rewards[0: len(ids)]:
                batch_rewards.append(reward)
            self._log_percentage(len(batch_rewards) / eval_episodes)

        self.evaluate = False
        self.env.modify_probs(batch_rewards)
        print(f"\nEvaluation over {eval_episodes} episodes: {np.mean(batch_rewards):.3f}")
        print("----------------------------------------------------------------------------")
        return  np.mean(batch_rewards), batch_rewards

    def _eval_episodes(self, ids):
        batch_turning_intentions, batch_observations, batch_terminated, batch_truncated, batch_rewards, batch_infos = self._batch_initialize_episode(ids)
        ep_steps = 0
        batch_score = [0 for _ in range(len(batch_observations))]
        while (not self._batch_is_episode_ended(batch_observations, batch_rewards, batch_terminated, batch_infos) and  ep_steps < 1000) or ep_steps < 1:
            
            batch_agent_actions = self._batch_select_actions(batch_turning_intentions, batch_observations, batch_terminated, batch_truncated)
            batch_observations, batch_rewards, batch_terminated, batch_truncated, batch_infos = self.env.step(
                [ 
                   {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()} for agent_actions in batch_agent_actions
                ]
            )
            
            batch_score = [sum(rewards) + score for rewards, score in zip(batch_rewards, batch_score)]
            ep_steps += 1
        return batch_score
    
    def _envision_episode(self, id):
        turning_intentions, observations, terminated, truncated, rewards, info = self._initialize_episode(id)
        print(turning_intentions)
        ep_steps = 0
        test = []
        test1 = []
        self.evaluate = True
        while (not self._is_episode_ended(observations, rewards, terminated, info)) or ep_steps < 1:
            
            agent_actions = self._select_actions(turning_intentions, observations, terminated, truncated)
            test.append(agent_actions)
            observations, rewards, terminated, truncated, info = self.single_env.step(
                {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()}
            )
            test1.append({agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()})
            ep_steps += 1
        self.evaluate = True
        print(f"Finished envisioning episode because ")
        return  test, test1

    
    def _full_eval_episodes(self, ids, data_collector: ExperimentDataCollector):
        batch_turning_intentions, batch_observations, batch_terminated, batch_truncated, batch_rewards, batch_infos = self._batch_initialize_episode(ids)
        ep_steps = 0
        data_collector.start_new_scenarios(ids, batch_turning_intentions)
        batch_score = [0 for _ in range(len(batch_observations))]
        print(f"Starting evaluation of {len(ids)} episodes")
        while (not self._batch_is_episode_ended(batch_observations, batch_rewards, batch_terminated, batch_infos) and  ep_steps < 1000) or ep_steps < 1:
            
            batch_agent_actions = self._batch_select_actions(batch_turning_intentions, batch_observations, batch_terminated, batch_truncated)
            batch_observations, batch_rewards, batch_terminated, batch_truncated, batch_infos = self.env.step(
                [ 
                   {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()} for agent_actions in batch_agent_actions
                ]
            )
            self._extract_scenario_data_batch(ids, batch_observations, batch_infos, data_collector)
            batch_score = [sum(rewards) + score for rewards, score in zip(batch_rewards, batch_score)]
            self._log_progress(np.average(batch_score), ep_steps)
            ep_steps += 1
        data_collector.close_scenario()
        return batch_score
    
    def _full_eval_episode(self, id, data_collector: ExperimentDataCollector):
        turning_intentions, observations, terminated, truncated, rewards, info = self._initialize_episode(id)
        ep_steps = 0
        data_collector.start_new_scenarios([id], turning_intentions)
        while (not self._is_episode_ended(observations, rewards, terminated, info) and ep_steps < 1000) or ep_steps < 1: 
            agent_actions = self._select_actions(turning_intentions, observations, terminated, truncated)
            observations, rewards, terminated, truncated, info = self.single_env.step(
                {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()}
            )
            self._extract_scenario_data(id, observations, info, data_collector)
            ep_steps += 1
        data_collector.close_scenario()
    
    def _extract_scenario_data_batch(self, ids, batch_observations, batch_infos, data_collector: ExperimentDataCollector):
        for id, observations, infos in zip(ids, batch_observations, batch_infos):
            self._extract_scenario_data(id, observations, infos, data_collector)

    def _extract_scenario_data(self, id, observations, infos, data_collector:Union[ExperimentDataCollector]):
        for agent_id in self.agent_names:
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
                    is_waiting=is_waiting,
                    scenario_id=id
                )
                if infos[agent_id]['env_obs'].events.collisions:
                    data_collector.mark_agent_crashed(agent_id, id)

                if infos[agent_id]['env_obs'].events.reached_goal:
                    data_collector.mark_agent_succeeded(agent_id, id)
        for social_traffic in infos["social_traffic"]:
            data_collector.add_social_vehicle(social_traffic["id"], id)
            data_collector.record_agent_data(
                    social_traffic["id"],
                    speed=social_traffic["speed"],
                    acceleration=social_traffic["linear_acceleration"][0],
                    dt=social_traffic["dt"],
                    travel_distance=social_traffic["travel_distance"],
                    is_waiting=(social_traffic["speed"] < 0.1),
                    scenario_id=id
                )

    def full_eval(self, parallel = True):
        eval_episodes = len(self.scenarios)
        self.evaluate = True

        if parallel:
            data_collector = ExperimentDataCollector(self.algorithm_identifier)
            for ids in self.slice_list(list(range(eval_episodes)), self.num_env):
                self._full_eval_episodes(ids, data_collector)
            data_collector.save_raw_data()
        else:
            data_collector = ExperimentDataCollector(self.algorithm_identifier)
            for id in range(eval_episodes):
                self._full_eval_episode(id, data_collector)
            data_collector.save_raw_data()
        print(f'Finished evaluation')
        self.evaluate = False