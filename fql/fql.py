from cvae import ContrastiveVAE

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def q1(self, state, action):
        return self.q1_net(torch.cat([state, action], dim=1))

class FQL:
    def __init__(self, state_dim, action_dim, max_action, device,
                 gamma=0.99, tau=0.005, policy_lr=3e-4, q_lr=1e-3, cvae_lr=1e-3, beta=2.0, use_tc=True, augmentation=False, policy_frequency=2):

        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.policy_frequency = policy_frequency
        self.augmentation = augmentation

        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

        self.vae = ContrastiveVAE(state_dim, action_dim, latent_dim, cvae_lr, max_action, beta, use_tc, device, use_bias=False).to(device)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0).repeat(128, 1)
            action = self.actor(state, self.vae.decode(state))
            q_values = self.critic.q1(state, action)
            best = q_values.argmax(0)
            return action[best].cpu().numpy().flatten()
        
    def select_action_h(self, state_h, action_h):
        with torch.no_grad():
            B, A, D = action_h.shape
            
            idx = self.critic.q1(
                state_h, action_h.reshape(B * A, D)
            ).view(B, A).argmin(dim=1)
            
            return action_h[torch.arange(B), idx]

    def train(self, replay_buffer, batch_size, global_step):
        state, action, next_state, reward, not_done, action_h = replay_buffer.sample(batch_size)
        
        B, A, D = action_h.shape
        state_h = state.unsqueeze(1).expand(-1, A, -1).reshape(B * A, -1)
        
        if self.augmentation:
            action_h = action_h.reshape(B * A, D)
            cvae_loss = self.vae.train_vae(state, state_h, action, action_h)
    
        else:
            action_h = self.select_action_h(state_h, action_h)
            cvae_loss = self.vae.train_vae(state, state, action, action_h)
            
        critic_loss, q1_loss, q2_loss, q1, q2 = self.train_critic(state, action, next_state, reward, not_done, batch_size)

        if global_step % self.policy_frequency == 0:
            actor_loss = self.train_actor(state)
            self.soft_update_targets()

            return {
                "cvae_loss": cvae_loss,
                "qf_loss": critic_loss,
                "qf1_loss": q1_loss,
                "qf1_values": q1,
                "qf2_loss": q2_loss,
                "qf2_values": q2,
                "actor_loss": actor_loss,
                }
        
        else :
            return {
                "cvae_loss": cvae_loss,
                "qf_loss": critic_loss,
                "qf1_loss": q1_loss,
                "qf1_values": q1,
                "qf2_loss": q2_loss,
                "qf2_values": q2,
            }

    def train_critic(self, state, action, next_state, reward, not_done, batch_size):
        with torch.no_grad():
            next_state = next_state.unsqueeze(1).expand(-1, 16, -1).reshape(-1, state.shape[1])
            target_q1, target_q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q.view(batch_size, -1).max(1)[0].unsqueeze(1)
            target = reward + not_done * self.gamma * target_q

        q1, q2 = self.critic(state, action)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), q1_loss.item(), q2_loss.item(), q1.mean().item(), q2.mean().item()

    def train_actor(self, state):
        actor_loss = -self.critic.q1(state, self.actor(state, self.vae.decode(state))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update_targets(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.vae.state_dict(), filename + "_vae")
        torch.save(self.vae.optimizer.state_dict(), filename + "_vae_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
        self.vae.load_state_dict(torch.load(filename + "_vae"))
        self.vae.optimizer.load_state_dict(torch.load(filename + "_vae_optimizer"))