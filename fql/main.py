import argparse
import gymnasium as gym
import numpy as np
import time
import os
import torch
import fql
import utils
import wandb
from torch.utils.tensorboard import SummaryWriter

def interact_with_environment(env, eval_envs, state_dim, action_dim, max_action, device, args):
    setting = f"{args.env_id}_{args.seed}"
    start_time = time.time()

    policy = fql.FQL(
        state_dim, action_dim, max_action, device,
        args.gamma, args.tau, args.policy_lr, args.q_lr, args.cvae_lr, args.beta, args.use_tc, args.augmentation, args.policy_frequency
    )

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=args.buffer_size, max_action=max_action)

    evaluations = []
    state, _ = env.reset(seed=args.seed)
    
    epeward, ep_timesteps, ep_num = 0, 0, 0

    for global_step in range(int(args.total_timesteps)):
        ep_timesteps += 1

        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            noise = np.random.normal(0, max_action * args.exploration_noise, size=action_dim)
            action = (policy.select_action(np.array(state)) + noise)
            action = action.clip(-max_action, max_action)

        next_state, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        done_bool = float(done) if ep_timesteps < env.spec.max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        epeward += reward
                    
        if "episode" in infos:       
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

        if done:
            print(f"Total T: {global_step} Episode Num: {ep_num+1} Episode T: {ep_timesteps} Reward: {epeward:.3f}")
            state, _ = env.reset(seed=args.seed)
            epeward, ep_timesteps, ep_num = 0, 0, ep_num + 1

        if global_step >= args.learning_starts and replay_buffer.size >= args.batch_size:
            losses = policy.train(replay_buffer, args.batch_size, global_step)

            if global_step % 1000 == 0:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                for k, v in losses.items():
                    writer.add_scalar(f"losses/{k}", v, global_step)

            if global_step % 10000 == 0:
                eval_return = eval_policy(policy, eval_envs)
                evaluations.append(eval_return)
                np.save(f"./results/performance_{setting}", evaluations)
                policy.save(f"./models/policy_{setting}")
                writer.add_scalar("Test/return", eval_return, global_step)
                writer.add_scalar("Steps", global_step, global_step)

    policy.save(f"./models/policy_{setting}")
    replay_buffer.save(f"./buffers/buffer_{setting}")

def eval_policy(policy, eval_envs, eval_episodes=10):
    
    states, _ = eval_envs.reset(seed=range(eval_episodes))
    dones = np.zeros(eval_episodes, dtype=bool)
    returns = np.zeros(eval_episodes)

    while not np.all(dones):
        actions = np.array([policy.select_action(state) for state in states])
        next_states, rewards, terminated, truncated, _ = eval_envs.step(actions)

        returns += rewards * (~dones)
        dones |= (terminated | truncated)
        states = next_states

    avg_reward = returns.mean()
    print("-------------------------------------------")
    print(f"[Eval] Avg. Reward over {eval_episodes} episodes: {avg_reward:.3f}")
    print("-------------------------------------------")
    return avg_reward

def make_env(env_name, seed):
    def thunk():
        env = gym.make(env_name)
        env.reset(seed=seed)
        env.action_space.seed(seed) 
        return env
    return thunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", default='fql_continuous_action')
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--learning_starts", type=int, default=25000)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--exploration_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--cvae_lr", type=float, default=1e-3)
    parser.add_argument("--use_tc", action="store_true")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--torch_deterministic", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb_project_name", default='cleanRL')
    args = parser.parse_args()

    if args.track:
        wandb.init(project="cleanRL", name=f"{args.env_id}_{args.seed}", config=vars(args), sync_tensorboard=True, save_code=True)
        writer = SummaryWriter(log_dir=f"runs/{args.env_id}/fql/{args.beta}/{'Augmentation' if args.augmentation else 'Critic'}/{args.seed+1}")
    
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)
    
    print(f"[Start] FQL | Env: {args.env_id} | Seed: {args.seed} | CUDA: {args.cuda_id}")

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./buffers", exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(args.seed) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    eval_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i) for i in range(10)])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    interact_with_environment(env, eval_envs, state_dim, action_dim, max_action, device, args)
    wandb.finish()