"""PPO with an asymmetric actor-critic, running observation normalization and KL-adaptive learning rate."""
import torch
from torch import nn


def mlp(n_in, hidden, n_out):
    layers = []
    for h in hidden:
        layers += [nn.Linear(n_in, h), nn.ELU()]
        n_in = h
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Running mean and variance (parallel Welford), saved with the weights; with `scale` fixed: x * scale."""

    def __init__(self, dim, clip=10.0, scale=None):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32) ** -2)
        self.register_buffer('count', torch.tensor(1e-4))
        self.clip = clip
        self.fixed = scale is not None

    def update(self, x):
        if self.fixed:
            return
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = m - self.mean, self.count + n
        self.var.copy_((self.var * self.count + v * n + delta ** 2 * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, critic_dim, act_dim, actor_hidden, critic_hidden, init_std, obs_scale=None, max_std=None):
        super().__init__()
        # Fixed actor input scales keep a fixed lr meaning the same policy change during training; with running
        # normalization every input has unit variance and Adam moves the policy several times faster per step.
        self.obs_norm, self.critic_norm = Normalizer(obs_dim, scale=obs_scale), Normalizer(critic_dim)
        self.actor = mlp(obs_dim, actor_hidden, act_dim)
        self.critic = mlp(critic_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_std)).log())
        # Upper bound of the action spread: a growing spread degrades the walking policy into standing still.
        self.max_log_std = float(torch.tensor(float(max_std)).log()) if max_std else None

    def std(self):
        return self.log_std.exp() if self.max_log_std is None else self.log_std.clamp(max=self.max_log_std).exp()

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(self.obs_norm(obs)), self.std())

    def value(self, cobs):
        return self.critic(self.critic_norm(cobs)).squeeze(-1)


class PPO:
    def __init__(self, ac, cfg):
        self.ac, self.cfg = ac, cfg
        self.lr = cfg['lr']
        self.opt = torch.optim.Adam(ac.parameters(), lr=self.lr)

    def gae(self, rew, val, done, last_val):
        g, lam = self.cfg['gamma'], self.cfg['lam']
        adv = torch.zeros_like(rew)
        last = torch.zeros_like(last_val)
        for t in reversed(range(rew.shape[0])):
            nxt = last_val if t == rew.shape[0] - 1 else val[t + 1]
            not_done = 1.0 - done[t]
            delta = rew[t] + g * nxt * not_done - val[t]
            last = delta + g * lam * not_done * last
            adv[t] = last
        return adv, adv + val

    def update(self, obs, cobs, act, logp_old, adv, ret):
        c = self.cfg
        n = obs.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {'kl': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        k = 0
        for _ in range(c['epochs']):
            perm = torch.randperm(n)
            for idx in perm.chunk(c['minibatches']):
                d = self.ac.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                with torch.no_grad():
                    log_ratio = logp - logp_old[idx]
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
                if c['desired_kl'] is None:  # fixed learning rate
                    pass
                elif kl > 2 * c['desired_kl']:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif kl < c['desired_kl'] / 2:
                    self.lr = min(1e-2, self.lr * 1.5)
                for group in self.opt.param_groups:
                    group['lr'] = self.lr
                ratio = (logp - logp_old[idx]).exp()
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - c['clip'], 1 + c['clip']) * a).mean()
                vl = (self.ac.value(cobs[idx]) - ret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + c['value_coef'] * vl - c['entropy_coef'] * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c['max_grad_norm'])
                self.opt.step()
                for key, v in (('kl', kl), ('policy_loss', pl.item()), ('value_loss', vl.item()), ('entropy', ent.item())):
                    stats[key] += v
                k += 1
        return {key: v / k for key, v in stats.items()}
