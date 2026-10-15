"""Native benchmarks, with an objective-preserving Autoencode length extension."""

import json
import math
import pickle
import select
import struct
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from popgym.core.deck import Deck
from popgym.envs.autoencode import AutoencodeEasy, Mode


class TaskUnavailable(RuntimeError):
    """Native simulator or its dependencies could not be started."""


class ShortDeck(Deck):
    """Sample a prefix of a shuffled original deck, preserving all four suits."""

    def __init__(self, length: int) -> None:
        super().__init__(math.ceil(length / 52))
        self.length = length
        self.num_cards = length

    def reset(
        self, shuffle: bool = True, rng: np.random.Generator | None = None
    ) -> None:
        self.num_cards = 52 * self.num_decks
        super().reset(shuffle=shuffle, rng=rng)
        self.idx = self.idx[: self.length]
        self.num_cards = self.deck_len = self.length


class LengthAutoencode(AutoencodeEasy):
    """Use upstream step/reward/observations unchanged on shorter shuffled decks."""

    def __init__(self, length: int) -> None:
        if not 2 <= length <= 156:
            raise ValueError("Autoencode sequence length must lie in [2,156]")
        super().__init__(num_decks=math.ceil(length / 52))
        if length != self.deck.num_cards:
            self.deck = ShortDeck(length)
            self.deck.add_players("system")
            self.max_episode_length = 2 * length - 1
            self.state_space = gym.spaces.Tuple(
                (
                    gym.spaces.MultiDiscrete([4] * length),
                    gym.spaces.Discrete(2),
                    gym.spaces.Box(0, 1, (1,)),
                )
            )


class AutoencodeTask:
    """Six one-hot observable features; oracle suits are labels only during Stage B."""

    action_kind = "discrete"
    action_dim = 4
    observation_shape = (6,)
    env_name = "popgym-AutoencodeEasy"

    def __init__(self, difficulty: int = 52, **_: object) -> None:
        self.env = LengthAutoencode(difficulty)
        self.difficulty = difficulty
        self.max_steps = self.env.max_episode_length
        self.observation: tuple[int, int] | None = None
        self.finished = True

    def encode(self, observation: tuple[int, int]) -> np.ndarray:
        return gym.spaces.flatten(self.env.observation_space, observation).astype(
            np.float32
        )

    def reset(self, seed: int) -> np.ndarray:
        self.observation, _ = self.env.reset(seed=seed)
        self.finished = False
        return self.encode(self.observation)

    def label(self) -> tuple[int, bool]:
        """Privileged expert action for preparation/evaluation, never policy inputs."""
        hand = self.env.deck["system"]
        return int(self.env.deck.suits_idx[hand[-1]]), self.env.mode == Mode.PLAY

    def event(self) -> str:
        return "recall" if self.env.mode == Mode.PLAY else "cue"

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self.finished:
            raise RuntimeError("Reset required before stepping a finished episode")
        value = np.asarray(action)
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Expected a scalar integer Autoencode action")
        action = int(value)
        if not self.env.action_space.contains(action):
            raise ValueError("Invalid Autoencode action")
        label, scored = self.label()
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.finished = bool(terminated or truncated)
        info.update(accuracy=float(action == label) if scored else None, success=None)
        return self.encode(self.observation), float(reward), self.finished, info

    def close(self) -> None:
        self.env.close()


class RememberColorTask:
    """Native RGB/continuous-control task in a separately provisioned interpreter.

    The simulator's published Python <3.12 / Torch 2.2 dependencies cannot be
    installed in the manager's pinned Python 3.12 environment. A local binary
    pipe transfers only observations, actions, native rewards and diagnostics.
    """

    action_kind = "continuous"
    action_dim = 8
    observation_shape = (6, 128, 128)
    env_name = "mikasa_robo_RememberColor3-v0"

    def __init__(
        self,
        difficulty: int = 5,
        native_python: str | None = None,
        sim_backend: str = "cpu",
        render_backend: str = "cpu",
        **_: object,
    ) -> None:
        if difficulty < 1:
            raise ValueError("RememberColor recall delay must be positive")
        if native_python is None:
            raise TaskUnavailable(
                "RememberColor3-v0 requires task.native_python pointing to a Python 3.11 MIKASA/ManiSkill environment; see step_3_implementation.md"
            )
        self.difficulty = difficulty
        self.max_steps = 60
        self.time = 0
        self.finished = True
        config = {
            "delta_time": difficulty,
            "sim_backend": sim_backend,
            "render_backend": render_backend,
        }
        try:
            self.process = subprocess.Popen(
                [
                    native_python,
                    str(Path(__file__).with_name("robot_worker.py")),
                    json.dumps(config),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        except OSError as exc:
            raise TaskUnavailable(
                f"Cannot start native robot interpreter: {exc}"
            ) from exc
        try:
            handshake = self._receive()
            if "error" in handshake:
                raise TaskUnavailable(handshake["error"])
            if tuple(handshake["action_shape"]) != (8,):
                raise TaskUnavailable(f"Unexpected robot action shape: {handshake}")
        except BaseException:
            self.close()
            raise

    def _receive(self) -> dict:
        stream = self.process.stdout
        if not select.select([stream], [], [], 60)[0]:
            raise TaskUnavailable("Native robot worker timed out")
        header = stream.read(8)
        if len(header) != 8:
            raise TaskUnavailable(
                "Native robot worker exited; inspect its stderr for dependency or renderer errors"
            )
        size = struct.unpack("!Q", header)[0]
        payload = bytearray()
        while len(payload) < size:
            part = stream.read(size - len(payload))
            if not part:
                raise TaskUnavailable("Native robot worker disconnected")
            payload.extend(part)
        reply = pickle.loads(payload)
        if "error" in reply:
            raise TaskUnavailable(reply["error"])
        return reply

    def _request(self, request: dict) -> dict:
        data = pickle.dumps(request, protocol=4)
        try:
            self.process.stdin.write(struct.pack("!Q", len(data)) + data)
            self.process.stdin.flush()
            return self._receive()
        except (OSError, EOFError) as exc:
            raise TaskUnavailable("Native robot worker disconnected") from exc

    def reset(self, seed: int) -> np.ndarray:
        reply = self._request({"command": "reset", "seed": seed})
        self.time = 0
        self.finished = False
        return reply["observation"]

    def label(self) -> tuple[np.ndarray, bool]:
        return np.zeros(8, dtype=np.float32), False

    def event(self) -> str:
        return (
            "cue"
            if self.time < 5
            else ("delay" if self.time < 5 + self.difficulty else "recall")
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self.finished:
            raise RuntimeError("Reset required before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (8,)
            or not np.isfinite(action).all()
            or (np.abs(action) > 1).any()
        ):
            raise ValueError("Expected bounded robot action [8]")
        reply = self._request({"command": "step", "action": action})
        self.time += 1
        self.finished = reply["done"]
        return reply["observation"], reply["reward"], self.finished, reply["info"]

    def close(self) -> None:
        if getattr(self, "process", None) is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()


def make_task(
    config: dict, difficulty: int | None = None
) -> AutoencodeTask | RememberColorTask:
    """Construct the requested native task; unavailable tasks never become toy tasks."""
    options = {k: v for k, v in config.items() if k not in {"name", "target"}}
    options["difficulty"] = config["target"] if difficulty is None else difficulty
    if config["name"] == "autoencode":
        return AutoencodeTask(**options)
    if config["name"] == "remember_color":
        return RememberColorTask(**options)
    raise ValueError("Expected autoencode or remember_color")
