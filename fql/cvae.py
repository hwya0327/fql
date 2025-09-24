import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6

class ContrastiveVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, lr, max_action, beta, use_tc, device, use_bias=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_action = max_action
        self.beta = beta
        self.use_tc = use_tc
        self.device = device
        self.use_bias = use_bias

        self.encoder_z = self._make_encoder(state_dim + action_dim, 512)
        self.encoder_s = self._make_encoder(state_dim + action_dim, 512)

        self.z_mean = nn.Linear(512, latent_dim, bias=self.use_bias)
        self.z_logstd = nn.Linear(512, latent_dim, bias=self.use_bias)
        self.s_mean = nn.Linear(512, latent_dim, bias=self.use_bias)
        self.s_logstd = nn.Linear(512, latent_dim, bias=self.use_bias)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + 2 * latent_dim, 512, bias=self.use_bias),
            nn.ReLU(),
            nn.Linear(512, 512, bias=self.use_bias),
            nn.ReLU(),
            nn.Linear(512, action_dim, bias=self.use_bias)
        )

        if use_tc:
            self.discriminator = nn.Sequential(
                nn.Linear(latent_dim * 2, 1),
                nn.Sigmoid()
            )
            self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _make_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=self.use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=self.use_bias),
            nn.ReLU()
        )

    def reparam(self, mu, logstd):
        std = logstd.exp()
        return mu + std * torch.randn_like(std)

    def encode_z(self, state, action):
        h = self.encoder_z(torch.cat([state, action], dim=1))
        mu = self.z_mean(h)
        logstd = self.z_logstd(h).clamp(-5, 2)
        return self.reparam(mu, logstd), mu, logstd

    def encode_s(self, state, action):
        h = self.encoder_s(torch.cat([state, action], dim=1))
        mu = self.s_mean(h)
        logstd = self.s_logstd(h).clamp(-5, 2)
        return self.reparam(mu, logstd), mu, logstd

    def decode(self, state, z=None, s=None):
        z = z if z is not None else torch.randn((state.shape[0], self.latent_dim), device=self.device).clamp(-1.0, 1.0)
        s = s if s is not None else torch.randn((state.shape[0], self.latent_dim), device=self.device).clamp(-1.0, 1.0)
        out = self.decoder(torch.cat([state, z, s], dim=1))
        return self.max_action * torch.tanh(out)

    def forward(self, state, state_h, action, action_h):
        
        z_t, mu_z_t, logstd_z_t = self.encode_z(state, action)
        s_t, mu_s_t, logstd_s_t = self.encode_s(state, action)
        s_h, mu_s_h, logstd_s_h = self.encode_s(state_h, action_h)

        recon_t = self.decode(state, z_t, s_t)
        recon_h = self.decode(state_h, torch.zeros_like(s_h), s_h)

        return {
            "z_t": z_t, "s_t": s_t,
            "recon_t": recon_t, "recon_h": recon_h,
            "mu_z_t": mu_z_t, "logstd_z_t": logstd_z_t,
            "mu_s_t": mu_s_t, "logstd_s_t": logstd_s_t,
            "mu_s_h": mu_s_h, "logstd_s_h": logstd_s_h
        }

    def train_vae(self, state, state_h, action, action_h):

        out = self.forward(state, state_h, action, action_h)

        rec_t = F.mse_loss(out["recon_t"], action, reduction="mean") * action.shape[1]
        rec_h = F.mse_loss(out["recon_h"], action_h, reduction="mean") * action.shape[1]

        def kl(mu, logstd):
            return 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1 - 2 * logstd).sum(dim=1).mean()

        kl_z = kl(out["mu_z_t"], out["logstd_z_t"])
        kl_s_t = kl(out["mu_s_t"], out["logstd_s_t"])
        kl_s_h = kl(out["mu_s_h"], out["logstd_s_h"])

        loss = rec_t + rec_h + self.beta * (kl_z + kl_s_t + kl_s_h)

        if self.use_tc:
            z, s = out["z_t"], out["s_t"]
            z1, z2 = z.chunk(2, dim=0)
            s1, s2 = s.chunk(2, dim=0)

            q_pos = torch.cat([torch.cat([s1, z1], dim=1), torch.cat([s2, z2], dim=1)], dim=0)
            q_neg = torch.cat([torch.cat([s1, z2], dim=1), torch.cat([s2, z1], dim=1)], dim=0)

            with torch.no_grad():
                d_pos = self.discriminator(q_pos).clamp(EPSILON, 1 - EPSILON)
            tc_loss = torch.log(d_pos / (1 - d_pos)).mean()

            d_pos_ = self.discriminator(q_pos.detach()).clamp(EPSILON, 1 - EPSILON)
            d_neg_ = self.discriminator(q_neg.detach()).clamp(EPSILON, 1 - EPSILON)
            disc_loss = -torch.log(d_pos_).mean() - torch.log(1 - d_neg_).mean()

            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            loss += tc_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
