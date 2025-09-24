import os
import argparse
import subprocess
import time
import threading

env_config = {
    "HalfCheetah-v4": {"total_timesteps": 5000000, "learning_starts": 25000, 'tau': 0.005, 'beta': 2.0, 'q_lr': 0.001},
    "Hopper-v4": {"total_timesteps": 5000000, "learning_starts": 5000, 'tau': 0.003, 'beta': 5.0, 'q_lr': 0.0003},
    "Walker2d-v4": {"total_timesteps": 5000000, "learning_starts": 10000, 'tau': 0.005, 'beta': 2.0, 'q_lr': 0.001},
    "Ant-v4": {"total_timesteps": 5000000, "learning_starts": 25000, 'tau': 0.005, 'beta': 0.0, 'q_lr': 0.001},
    "Humanoid-v4": {"total_timesteps": 5000000, "learning_starts": 25000, 'tau': 0.003, 'beta': 0.0, 'q_lr': 0.0003},
}

os.makedirs("logs", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("buffers", exist_ok=True)

def run_env_seeds(env, config, gpu_id):
    processes = []
    for seed in range(0,5):
        print(f"[GPU {gpu_id}] Launching | Env: {env} | Seed: {seed}")

        cmd = [
            "python", "main.py",
            "--env_id", env,
            "--seed", str(seed),
            "--total_timesteps", str(config["total_timesteps"]),
            "--learning_starts", str(config["learning_starts"]),
            "--policy_frequency", "2",
            "--exploration_noise", "0.1",
            "--batch_size", "256",
            "--gamma", "0.99",
            "--tau", str(config["tau"]),
            "--beta", str(config["beta"]),
            "--policy_lr", "0.0003",
            "--q_lr", str(config["q_lr"]),
            "--cvae_lr", "0.001",
            "--use_tc",
            '--track',
            '--torch_deterministic',
            "--cuda_id",  str(gpu_id),
        ]   

        log_path = f"logs/{env}_{seed}.log"
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
        processes.append((p, f))

    for p, f in processes:
        p.wait()
        f.close()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", type=int, required=True)
    args = parser.parse_args()

    start_time = time.time()
    print(f"--- Experiment Start : {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    threads = []
    for env, config in env_config.items():
        t = threading.Thread(target=run_env_seeds, args=(env, config, args.cuda_id))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    end_time = time.time()
    total_execution_time = end_time - start_time
    
    hours = int(total_execution_time // 3600)
    minutes = int((total_execution_time % 3600) // 60)
    seconds = int(total_execution_time % 60)

    print(f"--- Experiment End : {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- Total Excution Time : {hours}h {minutes}m {seconds}s ---")