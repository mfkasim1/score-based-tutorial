import torch

# generate the dataset
xtns = torch.randn((1000, 2)) * 0.2 + 0.3
dset = torch.utils.data.TensorDataset(xtns)

class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(2 * x)

class ScoreNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logp_network = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            Sine(),
            torch.nn.Linear(64, 64),
            torch.nn.LogSigmoid(),
            torch.nn.Linear(64, 64),
            torch.nn.LogSigmoid(),
            torch.nn.Linear(64, 1),
        )
    def logp(self, x, t):
        xt = torch.cat((x, t), dim=-1)
        logp = self.logp_network(xt)
        return logp
    def forward(self, x, t):
        x = x.requires_grad_()
        with torch.enable_grad():
            logp = self.logp(x, t)
        score = torch.autograd.grad(logp, x, grad_outputs=torch.ones_like(logp), create_graph=torch.is_grad_enabled(), retain_graph=True)[0]
        return score

score_network = ScoreNet()

def calc_loss(score_network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, 2) is the training data
    # sample the time
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4
    # calculate the terms for the posterior log distribution
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t  # integral of beta
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t  # (batch_size, 2)
    # calculate the score function
    score = score_network(x_t, t)  # score: (batch_size, 2)
    # calculate the loss function
    loss = (score - grad_log_p) ** 2
    lmbda_t = var_t
    weighted_loss = lmbda_t * loss
    return torch.mean(weighted_loss)

# start the training loop
from tqdm import tqdm
opt = torch.optim.Adam(score_network.parameters(), lr=3e-4)
dloader = torch.utils.data.DataLoader(dset, batch_size=256, shuffle=True)
for i_epoch in tqdm(range(150000)):
    for data, in dloader:
        # training step
        opt.zero_grad()
        loss = calc_loss(score_network, data)
        loss.backward()
        opt.step()

from typing import List

def generate_samples(score_network: torch.nn.Module, nsamples: int) -> List[torch.Tensor]:
    x_t = torch.randn((nsamples, 2))  # (nsamples, 2)
    time_pts = torch.linspace(1, 0, 1000)  # (ntime_pts,)
    res = [x_t]
    beta = lambda t: 0.1 + (20 - 0.1) * t
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        # calculate the drift and diffusion terms
        fxt = -0.5 * beta(t) * x_t
        gt = beta(t) ** 0.5
        score = score_network(x_t, t.expand(x_t.shape[0], 1)).detach()
        drift = fxt - gt * gt * score
        diffusion = gt
        # euler-maruyama step
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
        res.append(x_t)
    return res

samples = generate_samples(score_network, 1000)

from celluloid import Camera
import matplotlib.pyplot as plt
fig = plt.figure()
camera = Camera(fig)
ns = 100
bound = 2.0
xraw, yraw = torch.linspace(-bound, bound, ns), torch.linspace(-bound, bound, ns)
x, y = torch.meshgrid(xraw, yraw, indexing="xy")
xy = torch.stack((x, y), dim=-1).reshape(-1, 2)
time_pts = torch.linspace(1, 0, 1000)  # (ntime_pts,)
for i, sample in tqdm(enumerate(samples)):
    if (i > 500 and i % 10 == 0) or (i > 800 and i % 5 == 0):
        t = time_pts[i].expand((xy.shape[0], 1))
        uv = score_network(xy, t).reshape((ns, ns, 2)).detach()
        logp = score_network.logp(xy, t).reshape((ns, ns)).detach()
        _ = plt.imshow(torch.exp(logp).detach().numpy(), extent=(-bound, bound, bound, -bound), cmap="Oranges")
        _ = plt.streamplot(xraw.numpy(), yraw.numpy(), uv[..., 0].numpy(), uv[..., 1].numpy(), color='C0')
        _ = plt.plot(sample[:, 0].detach().cpu().numpy(), sample[:, 1].detach().cpu().numpy(), 'C1.')
        _ = plt.xlim(-bound, bound)
        _ = plt.ylim(-bound, bound)
        _ = camera.snap()

for j in range(10):
    _ = plt.imshow(torch.exp(logp).detach().numpy(), extent=(-bound, bound, bound, -bound), cmap="Oranges")
    _ = plt.streamplot(xraw.numpy(), yraw.numpy(), uv[..., 0].numpy(), uv[..., 1].numpy(), color='C0')
    _ = plt.plot(sample[:, 0].detach().cpu().numpy(), sample[:, 1].detach().cpu().numpy(), 'C1.')
    _ = plt.xlim(-bound, bound)
    _ = plt.ylim(-bound, bound)
    _ = camera.snap()

animation = camera.animate()
animation.save('02-animation-sampling-anim-sde.mp4')
