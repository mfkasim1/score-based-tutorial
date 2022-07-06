import torch

# generate the swiss roll dataset
xtns = torch.randn((1000, 2)) * 0.2 + 0.3
dset = torch.utils.data.TensorDataset(xtns)

# score_network takes input of 2 dimension and returns the output of the same size
class ScoreNet(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 2)
        v = -(x - 0.3) / 0.2 ** 2
        return v

score_network = ScoreNet()

from functorch import jacrev, vmap

def calc_loss(score_network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, 2) is the training data
    score = score_network(x)  # score: (batch_size, 2)
    # first term: half of the squared norm
    term1 = torch.linalg.norm(score, dim=-1) ** 2 * 0.5
    # second term: trace of the Jacobian
    jac = vmap(jacrev(score_network))(x)  # (batch_size, 2, 2)
    term2 = torch.einsum("bii->b", jac)
    return (term1 + term2).mean()

from tqdm import tqdm

def generate_samples(score_net: torch.nn.Module, nsamples: int, eps: float = 0.001, nsteps: int = 1000):
    # generate samples using Langevin MCMC
    # x0: (sample_size, nch)
    x0 = torch.rand((nsamples, 2)) * 2 - 1
    xs = [x0]
    for i in range(nsteps):
        z = torch.randn_like(x0)
        x0 = x0 + eps * score_net(x0) + (2 * eps) ** 0.5 * z
        xs.append(x0)
    return xs

score_network = score_network.to(torch.device("cpu"))
samples = generate_samples(score_network, 1000)

from celluloid import Camera
import matplotlib.pyplot as plt
fig = plt.figure()
camera = Camera(fig)
ns = 100
xraw, yraw = torch.linspace(-1, 1, ns), torch.linspace(-1, 1, ns)
x, y = torch.meshgrid(xraw, yraw, indexing="xy")
xy = torch.stack((x, y), dim=-1).reshape(-1, 2)
uv = score_network(xy).reshape((ns, ns, 2)).detach()
true_logp = torch.exp(-torch.sum((xy - 0.3) ** 2, dim=-1) / (2 * 0.2 ** 2)).reshape((ns, ns))
for i, sample in tqdm(enumerate(samples)):
    if i < 100:
        _ = plt.imshow(true_logp.detach().numpy(), extent=(-1, 1, 1, -1), cmap="Oranges")
        _ = plt.streamplot(xraw.numpy(), yraw.numpy(), uv[..., 0].numpy(), uv[..., 1].numpy(), color='C0')
        _ = plt.plot(sample[:, 0].detach().cpu().numpy(), sample[:, 1].detach().cpu().numpy(), 'C1.')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        camera.snap()

animation = camera.animate()
animation.save('02-animation-sampling-wrong.mp4')
