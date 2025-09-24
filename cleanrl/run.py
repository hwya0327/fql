import os
import subprocess
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.chdir(base_dir)

algorithms = [
    "sac_continuous_action",
    "meow_continuous_action",
    "td3_continuous_action",
    "ddpg_continuous_action",
    "ppo_continuous_action"
]

env_timesteps = {
    "HalfCheetah-v4": 5000000,
    "Hopper-v4": 5000000,
    "Walker2d-v4": 5000000,
    "Ant-v4": 5000000,
    "Humanoid-v4": 5000000,
}

meow_params = {
    "HalfCheetah-v4": {
        "alpha": 0.25, "tau": 0.003,
        "sigma_max": 2.0, "sigma_min": -5.0,
        "learning_starts": 10000, "deterministic_action": False
    },
    "Hopper-v4": {
        "alpha": 0.25, "tau": 0.005,
        "sigma_max": -0.3, "sigma_min": -5.0,
        "learning_starts": 5000, "deterministic_action": False
    },
    "Walker2d-v4": {
        "alpha": 0.1,  "tau": 0.005,
        "sigma_max": -0.3, "sigma_min": -5.0,
        "learning_starts": 10000, "deterministic_action": False
    },
    "Ant-v4": {
        "alpha": 0.05, "tau": 0.0001,
        "sigma_max": -0.3, "sigma_min": -5.0,
        "learning_starts": 5000, "deterministic_action": False
    },
    "Humanoid-v4": {
        "alpha": 0.125, "tau": 0.0005,
        "sigma_max": -0.3,   "sigma_min": -5.0,
        "learning_starts": 5000, "deterministic_action": False
    },
}

os.makedirs("logs", exist_ok=True)

for algo in algorithms:
    print(f"\n🚀 Starting all environments for algorithm: {algo}")
    processes = []
    log_files = []

    for env_id, timesteps in env_timesteps.items():
        for seed in range(5):
            log_path = f"logs/{algo}_{env_id}_{seed}.log"
            log_file = open(log_path, "w")
            log_files.append(log_file)

            cmd = [
                "poetry", "run", "python", f"cleanrl/{algo}.py",
                "--env-id", env_id,
                "--seed", str(seed),
                "--total-timesteps", str(timesteps),
                "--track"
            ]

            if algo == "meow_continuous_action":
                params = meow_params.get(env_id, {})
                for key, val in params.items():
                    flag = f"--{key.replace('_', '-')}"
                    cmd += [flag, str(val)]

            print(f"  > Launching {algo} | {env_id} | seed {seed}")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append(p)

    for p, log in zip(processes, log_files):
        p.wait()
        log.close()

    print(f"\n✅ Finished ALL environments for algorithm: {algo}")
    print("=" * 60)
