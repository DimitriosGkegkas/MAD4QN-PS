# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import multiprocessing as mp
import sys
import traceback
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Tuple

import cloudpickle
import gymnasium as gym


EnvConstructor = Callable[[int], gym.Env]



class _Message(Enum):
    SEED = 1
    ACCESS = 2
    RESET = 3
    STEP = 4
    RESULT = 5
    CLOSE = 6
    EXCEPTION = 7
    SCENARIO = 8
    PROBS = 9


class ParallelEnvWithScenario(object):
    """Batch together multiple environments and step them in parallel. Each environment
    is simulated in an external process for lock-free parallelism using `multiprocessing`
    processes, and pipes for communication.

    Note:
        Simulation might slow down when number of parallel environments requested
        exceed number of available CPUs.
    """

    def __init__(
        self,
        env_constructors: Sequence[EnvConstructor],
        auto_reset: bool,
        seed: int = 42,
    ):
        """The environments can be different but must use the same action and
        observation spaces.

        Args:
            env_constructors (Sequence[EnvConstructor]): List of callables that create environments.
            auto_reset (bool): Automatically resets an environment when episode ends.
            seed (int, optional): Seed for the first environment. Defaults to 42.

        Raises:
            TypeError: If any environment constructor is not callable.
            ValueError: If the action or observation spaces do not match.
        """

        if len(env_constructors) > mp.cpu_count():
            warnings.warn(
                f"Simulation might slow down, as the requested number of parallel "
                f"environments ({len(env_constructors)}) exceed the number of available "
                f"CPUs ({mp.cpu_count()}).",
                ResourceWarning,
            )

        if any([not callable(ctor) for ctor in env_constructors]):
            raise TypeError(
                f"Found non-callable `env_constructors`. Expected `env_constructors` of type "
                f"`Sequence[Callable[[int], gym.Env]]`, but got {env_constructors})."
            )

        self._num_envs = len(env_constructors)
        self._polling_period = 0.1
        self._closed = False

        # Fork is not a thread safe method.
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        mp_ctx = mp.get_context(start_method)

        self._parent_pipes = []
        self._processes = []
        for idx, env_constructor in enumerate(env_constructors):
            cur_seed = seed + idx
            parent_pipe, child_pipe = mp_ctx.Pipe()
            process = mp_ctx.Process(
                target=_worker,
                name=f"Worker-<{type(self).__name__}>-<{idx}>",
                args=(
                    cloudpickle.dumps(env_constructor),
                    cur_seed,
                    auto_reset,
                    child_pipe,
                    self._polling_period,
                ),
            )
            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)

            # Daemonic subprocesses quit when parent process quits. However, daemonic
            # processes cannot spawn children. Hence, `process.daemon` is set to False.
            process.daemon = False
            process.start()
            child_pipe.close()

        self._wait_start()
        self._single_observation_space, self._single_action_space = self._get_spaces()

    @property
    def batch_size(self) -> int:
        """The number of environments."""
        return self._num_envs

    @property
    def observation_space(self) -> gym.Space:
        """The environment's observation space in gym representation."""
        return self._single_observation_space

    @property
    def action_space(self) -> gym.Space:
        """The environment's action space in gym representation."""
        return self._single_action_space

    def _call(self, msg: _Message, payloads: Sequence[Any]) -> Sequence[Any]:
        assert len(payloads) == self._num_envs
        for pipe, payload in zip(self._parent_pipes, payloads):
            pipe.send((msg, payload))

        return self._recv()

    def _recv(self) -> Sequence[Any]:
        messages = []
        payloads = []
        for pipe in self._parent_pipes:
            message, payload = pipe.recv()
            messages.append(message)
            payloads.append(payload)

        for messages, payload in zip(messages, payloads):
            if message == _Message.EXCEPTION:
                worker_name, stacktrace = payload
                self.close()
                raise Exception(f"\n{worker_name}\n{stacktrace}")

        return payloads

    def _wait_start(self):
        self._recv()

    def _get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        observation_spaces = self._call(
            _Message.ACCESS, ["observation_space"] * self._num_envs
        )
        observation_space = observation_spaces[0]
        if any([space != observation_space for space in observation_spaces]):
            raise ValueError(
                f"Expected all environments to have the same observation space, "
                f"but got {observation_spaces}."
            )

        action_spaces = self._call(_Message.ACCESS, ["action_space"] * self._num_envs)
        action_space = action_spaces[0]
        if any([space != action_space for space in action_spaces]):
            raise ValueError(
                f"Expected all environments to have the same action space, "
                f"but got {action_spaces}."
            )

        return observation_space, action_space
    
    def set_scenario(self, scenarios: Sequence[str]):
        """Sets the scenario for each environment.

        Args:
            scenarios (Sequence[str]): List of scenarios for each environment.
        """

        # add -1 in order to fill the missing scenarios        
        if len(scenarios) != self._num_envs:
            scenarios += ["-1"] * (self._num_envs - len(scenarios))
        self._call(_Message.SCENARIO, scenarios)
    
    def modify_probs(self, probs: Sequence[float]):
        """Sets the probabilities for each environment.

        Args:
            probs (Sequence[float]): List of probabilities for each environment.
        """
        self._call(_Message.PROBS, [probs] * self._num_envs)


    def seed(self) -> Sequence[int]:
        """Retrieves the seed used in each environment.

        Returns:
            Sequence[int]: Seed of each environment.
        """
        seeds = self._call(_Message.SEED, [None] * self._num_envs)
        return seeds

    def reset(self) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
        """Reset all environments.

        Returns:
            Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]: A batch of
                observations and infos from the vectorized environment.
        """

        # since the return is [(obs0, infos0), ...] they need to be zipped to form.
        #   [(obs0, ...), (infos0, ...)]
        observations, infos = zip(*self._call(_Message.RESET, [None] * self._num_envs))
        return observations, infos

    def step(
        self, actions: Sequence[Dict[str, Any]]
    ) -> Tuple[
        Sequence[Dict[str, Any]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, bool]],
        Sequence[Dict[str, bool]],
        Sequence[Dict[str, Any]],
    ]:
        """Steps all environments.

        Args:
            actions (Sequence[Dict[str,Any]]): Actions for each environment.

        Returns:
            Tuple[ Sequence[Dict[str, Any]], Sequence[Dict[str, float]], Sequence[Dict[str, bool]], Sequence[Dict[str, bool]], Sequence[Dict[str, Any]] ]:
                A batch of (observations, rewards, terminateds, truncateds, infos) from the vectorized environment.
        """
        result = self._call(_Message.STEP, actions)
        observations, rewards, terminateds, truncateds, infos = zip(*result)
        return (observations, rewards, terminateds, truncateds, infos)

    def close(self, terminate=False):
        """Sends a close message to all external processes.

        Args:
            terminate (bool, optional): If `True`, then the `close` operation is
                forced and all processes are terminated. Defaults to False.
        """

        if terminate:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self._parent_pipes:
                try:
                    pipe.send((_Message.CLOSE, None))
                    pipe.close()
                except IOError:
                    # The connection was already closed.
                    pass

        for process in self._processes:
            if process.is_alive():
                process.join()

        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close(terminate=True)



def _worker(
    env_constructor: bytes,
    seed: int,
    auto_reset: bool,
    pipe: mp.connection.Connection,
    polling_period: float = 0.1,
):
    """Process to build and run an environment. Using a pipe to
    communicate with parent, the process receives action, steps
    the environment, and returns the observations.

    Args:
        env_constructor (bytes): Cloudpickled callable which constructs the environment.
        seed (int): Seed for the environment.
        auto_reset (bool): If True, auto resets environment when episode ends.
        pipe (mp.connection.Connection): Child's end of the pipe.
        polling_period (float, optional): Time to wait for keyboard interrupts. Defaults to 0.1.

    Raises:
        KeyError: If unknown message type is received.
    """
    env = cloudpickle.loads(env_constructor)(seed=seed)
    pipe.send((_Message.RESULT, None))

    try:
        while True:
            if not pipe.poll(polling_period):
                continue
            message, payload = pipe.recv()
            if message == _Message.SEED:
                env_seed = env.seed
                pipe.send((_Message.RESULT, env_seed))
            elif message == _Message.ACCESS:
                result = getattr(env, payload, None)
                pipe.send((_Message.RESULT, result))
            elif message == _Message.RESET:
                observation, info = env.reset()
                pipe.send((_Message.RESULT, (observation, info)))
            elif message == _Message.STEP:
                observation, reward, terminated, truncated, info = env.step(payload)
                # TODO at some point I can check if there are less than 4 cars and end the simulation (reset it) to move on if I want oto have async.
                if terminated["__all__"] and auto_reset:
                    # Final observation can be obtained from `info` as follows:
                    # `final_obs = info[agent_id]["env_obs"]`
                    observation, _ = env.reset()
                pipe.send(
                    (
                        _Message.RESULT,
                        (observation, reward, terminated, truncated, info),
                    )
                )
            elif message == _Message.SCENARIO:
                env.set_scenario(payload)
                pipe.send((_Message.RESULT, None))
            elif message == _Message.PROBS:
                env.modify_probs(payload)
                pipe.send((_Message.RESULT, None))
            elif message == _Message.CLOSE:
                break
            else:
                raise KeyError(
                    f"Expected message from {_Message.__members__}, but got unknown message `{message}`."
                )
    except (Exception, KeyboardInterrupt):
        etype, evalue, tb = sys.exc_info()
        if etype == KeyboardInterrupt:
            stacktrace = "".join(traceback.format_exception(etype, evalue, None))
        else:
            stacktrace = "".join(traceback.format_exception(etype, evalue, tb))
        payload = (mp.current_process().name, stacktrace)
        pipe.send((_Message.EXCEPTION, payload))
    finally:
        env.close()
        pipe.close()