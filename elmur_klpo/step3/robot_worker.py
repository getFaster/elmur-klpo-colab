"""Standalone Python 3.11 native MIKASA worker; no imports from the training package."""

import contextlib
import importlib
import json
import os
import pickle
import struct
import sys
import traceback


def main() -> None:
    """Serve a trusted parent process over length-prefixed local pickle messages."""
    output = sys.stdout.buffer
    input_stream = sys.stdin.buffer
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/elmur-klpo-native-matplotlib")

    def send(value: dict) -> None:
        data = pickle.dumps(value, protocol=4)
        output.write(struct.pack("!Q", len(data)) + data)
        output.flush()

    # Simulator logging must never corrupt the binary protocol.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            import gymnasium as gym
            import numpy as np
            import torch

            torch.set_num_threads(1)

            try:
                importlib.import_module(
                    "mikasa_robo_suite.rl.memory_envs.remember_color"
                )
            except ModuleNotFoundError as exc:
                if exc.name not in {
                    "mikasa_robo_suite.rl",
                    "mikasa_robo_suite.rl.memory_envs",
                    "mikasa_robo_suite.rl.memory_envs.remember_color",
                }:
                    raise
                importlib.import_module("mikasa_robo_suite.remember_color")
            config = json.loads(sys.argv[1])
            env = gym.make(
                "RememberColor3-v0",
                num_envs=1,
                obs_mode="rgb",
                render_mode="rgb_array",
                control_mode="pd_joint_delta_pos",
                reward_mode="normalized_dense",
                delta_time=config["delta_time"],
                sim_backend=config["sim_backend"],
                render_backend=config["render_backend"],
            )
            space = getattr(env.unwrapped, "single_action_space", env.action_space)
            send(
                {
                    "action_shape": space.shape,
                    "python": sys.version,
                    "torch": torch.__version__,
                }
            )

            def scalar(value: object) -> object:
                return (
                    value.detach().cpu().reshape(-1)[0].item()
                    if isinstance(value, torch.Tensor)
                    else value
                )

            def observation(value: dict) -> np.ndarray:
                cameras = value["sensor_data"]
                # Match upstream concatenation order, with no oracle/proprioceptive fields.
                rgb = torch.cat([camera["rgb"] for camera in cameras.values()], -1)
                array = rgb[0].permute(2, 0, 1).float().cpu().numpy()
                if array.shape != (6, 128, 128):
                    raise ValueError(
                        f"Expected two native 128x128 RGB cameras, got {array.shape}"
                    )
                return array

            time = 0
            while True:
                header = input_stream.read(8)
                if not header:
                    break
                request = pickle.loads(
                    input_stream.read(struct.unpack("!Q", header)[0])
                )
                if request["command"] == "reset":
                    obs, _ = env.reset(seed=request["seed"])
                    time = 0
                    send({"observation": observation(obs)})
                else:
                    # RATE's env_info uses noop_steps=1, hence its wrapper's
                    # n_initial_steps is zero and task actions pass unchanged.
                    action = request["action"]
                    obs, reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device)[None]
                    )
                    time += 1
                    send(
                        {
                            "observation": observation(obs),
                            "reward": float(scalar(reward)),
                            "done": bool(scalar(terminated) or scalar(truncated)),
                            "info": {
                                "success": bool(scalar(info["success"])),
                                "accuracy": None,
                            },
                        }
                    )
            env.close()
        except Exception:
            send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
