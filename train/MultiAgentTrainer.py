import pathlib
import numpy as np
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch
from util_rgb import make_env, position2road, roads2t_i
from datetime import datetime
import sys
import os



class MultiAgentTrainer:
    def __init__(
        self,
        args,
        batch_size=256,
        best_score=-1000.0,
        total_steps=int(1e6),
        agent_count=4,
        algorithm_identifier='DuelingDDQNAgents',
        evaluation_step=5000,
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
        
        self.algorithm_identifier = algorithm_identifier
        self.evaluate = False
        self.readyForEvaluation = False

    def initialize_environment(self, agent_spec, scenario_subdir="scenarios/sumo/multi_scenario"):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.agent_names = [f"Agent-{i}" for i in range(self.agent_count)]
        agent_interfaces = {agent_id: agent_spec.interface for agent_id in self.agent_names}

        scenarios_path = pathlib.Path(__file__).absolute().parent.parent / scenario_subdir
        self.scenarios = [str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()]
        self.scenarios.sort()

        self.env = make_env("smarts.env:hiway-v1", agent_interfaces, self.scenarios, self.args.headless, self.args.seed)

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
        base_dir='models',
    ):
        mem_size = 1 if self.args.load_checkpoint else 100000
        chkpt_dir = os.path.join(
            base_dir, self.algorithm_identifier, datetime.now().strftime("%d%m%Y")
        )
        os.makedirs(chkpt_dir, exist_ok=True)

        input_dims = self.env.observation_space.shape
        agent_params = {
            'gamma': gamma,
            'epsilon': epsilon,
            'lr': lr,
            'input_dims': input_dims,
            'n_actions': 2,
            'eps_min': eps_min,
            'batch_size': batch_size,
            'replace': replace,
            'eps_dec': eps_dec,
            'chkpt_dir': chkpt_dir,
            'algo': self.algorithm_identifier,
            'mem_size': int(mem_size * mem_size_factor),
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

        if self.args.load_checkpoint:
            for agent in self.agents.values():
                agent.load_models()


    def train(self):
        while self.n_steps < self.total_steps:
            self._run_episode()

    def _run_episode(self):
        turning_intentions, observations, terminated, truncated, rewards, infos = self._initialize_episode()
        score, ep_steps = 0, 0
        while not self._is_episode_ended(turning_intentions, observations, rewards, infos, ep_steps):
            agent_actions = self._select_actions(turning_intentions, observations, terminated, truncated)
            observations_, rewards, terminated, truncated, infos = self.env.step(
                {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()}
            )
            score += sum(rewards)
            filtered_agents = [agent_name for agent_name in self.agent_names if agent_name in observations_]
            agent_rewards = {agent_name: rewards[idx] for idx, agent_name in enumerate(filtered_agents)}

            self._store_transitions(observations, agent_actions, agent_rewards, observations_, terminated, truncated, turning_intentions)
            self._update_agents()
            observations = observations_
            self.n_steps += 1
            ep_steps += 1
            self._log_progress(score, ep_steps)
            if self.n_steps % self.evaluation_step == 0:
                self.readyForEvaluation = True
        self._evaluate_if_needed()
        self.n_episodes += 1
        

    def _initialize_episode(self, id = None):
        turning_intentions = {}
        if id is not None:
            self.env.set_scenario(id)
        observations, infos = self.env.reset()
        for k in self.agent_names:
            start = position2road([infos[k]['env_obs'][5].mission.start.position.x, infos[k]['env_obs'][5].mission.start.position.y])
            goal = position2road([infos[k]['env_obs'][5].mission.goal.position.x, infos[k]['env_obs'][5].mission.goal.position.y])
            turning_intentions[k] = roads2t_i[start + goal]
        terminated, truncated = {agent_id: False for agent_id in self.agent_names}, {agent_id: False for agent_id in self.agent_names}
        rewards = [0 for _ in range(len(self.agent_names))]
        return turning_intentions, observations, terminated, truncated, rewards, infos

    def _is_episode_ended(self, turning_intentions, observations, rewards, infos, ep_steps):
        return len(observations) == 0 or ep_steps >= 1000 or all(reward == -1 for reward in rewards)
    
    def act(self, obs, turning_intention):
        if turning_intention == 'straight':
            return self.agents['straight'].choose_action(obs, self.evaluate)
        elif turning_intention == 'left':
            return self.agents['left'].choose_action(obs, self.evaluate)
        elif turning_intention == 'right':
            return self.agents['right'].choose_action(obs, self.evaluate)
        
    def format_action(self, action):
        return action

    def _select_actions(self, turning_intentions, observations, terminated, truncated):
        agent_actions = {}
        for idx, agent_name in enumerate(self.agent_names):
            if agent_name in observations and not terminated[agent_name] and not truncated[agent_name]:
                agent_actions[agent_name] = self.act(observations[agent_name], turning_intentions[agent_name])
        return agent_actions 

    def _store_transitions(self, observations, agent_actions, agent_rewards, observations_, terminated, truncated, turning_intentions):
        for idx, agent_name in enumerate(self.agent_names):
            if agent_name in observations and agent_name in observations_:
                done = terminated[agent_name] or truncated[agent_name]
                self.agents[turning_intentions[agent_name]].store_transition(observations[agent_name], agent_actions[agent_name], agent_rewards[agent_name], observations_[agent_name], done)

    def _update_agents(self):
        # if self.n_steps % (self.batch_size // 2) == 0:
        for agent in self.agents.values():
            agent.learn()

    def _evaluate_if_needed(self):
        if self.readyForEvaluation:
            self.readyForEvaluation = False
            scores, scores_per_scenario = self.eval()
            elapsed_time = datetime.now() - self.start_time
            self.scores_list.append((scores, str(elapsed_time)))
            self.scores_per_scenario_list.append(scores_per_scenario)
            if scores > self.best_score:
                for agent in self.agents.values():
                    agent.save_models()
                self.best_score = scores

    def _log_progress(self, score, ep_steps):
        elapsed_time = datetime.now() - self.start_time
        sys.stdout.write(
            f"\r Total Steps: {self.n_steps} | Episode Step: {ep_steps} | Reward: {score:.2f} | Episode {self.n_episodes} | Elapsed Time: {elapsed_time}"
        )
        sys.stdout.flush()

    def _log_progress_eval(self, scenario, scenarios):
        elapsed_time = datetime.now() - self.start_time
        sys.stdout.write(
            f"\r Eval Scenario {scenario + 1}/{scenarios} | Elapsed Time: {elapsed_time}"
        )
        sys.stdout.flush()

    def eval(self):
        eval_episodes = len(self.scenarios)
        self.evaluate = True
        rewards = []
        for id in range(eval_episodes):
            self._log_progress_eval(id, eval_episodes)
            reward = self._eval_episode(id)
            rewards.append(reward)

        self.evaluate = False
        self.env.modify_probs(rewards)
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {np.mean(rewards):.3f}")
        print("---------------------------------------")
        return  np.mean(rewards), rewards




    def _eval_episode(self, id):
        turning_intentions, observations, terminated, truncated, rewards, infos = self._initialize_episode(id)
        score, ep_steps = 0, 0
        while not self._is_episode_ended(turning_intentions, observations, rewards, infos, ep_steps):
            
            agent_actions = self._select_actions(turning_intentions, observations, terminated, truncated)
            observations_, rewards, terminated, truncated, infos = self.env.step(
                {agent_name: self.format_action(agent_action) for agent_name, agent_action in agent_actions.items()}
            )
            score += sum(rewards)
            filtered_agents = [agent_name for agent_name in self.agent_names if agent_name in observations_]
            agent_rewards = {agent_name: rewards[idx] for idx, agent_name in enumerate(filtered_agents)}
            observations = observations_
        return score