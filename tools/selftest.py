"""Meaningful offline correctness checks; synthetic inputs are NOT accuracy evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.common import torch, np, seed_everything, save_json, ROOT
from lab.models import FaceCNN, ResNet18, VAE, UNet, Generator, Discriminator
from lab.part1 import square_wave_fourier, naive_dft_torch
from lab.mri import vae_loss, segmentation_loss, dice_from_confusion
from lab.cifar import augment_batch


def main():
    seed_everything()
    checks = []
    def passed(name):
        checks.append(name)
        print("PASS:", name, flush=True)
    t = torch.arange(256, dtype=torch.float64) / 256
    signal = square_wave_fourier(t)
    actual = naive_dft_torch(signal).numpy()
    assert np.allclose(actual, np.fft.fft(signal.numpy()), atol=1e-7)
    amp = abs(actual[:129]) / 128
    assert np.allclose(amp[np.arange(1, 100, 2)], 4 / (np.pi * np.arange(1, 100, 2)))
    passed("Explicit tensor DFT matches FFT and Fourier coefficients")
    face = FaceCNN(7)
    convs = [m for m in face.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 2 and all(m.out_channels == 32 and m.kernel_size == (3, 3) for m in convs)
    logits = face(torch.rand(2, 1, 50, 37))
    assert logits.shape == (2, 7)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    passed("Required LFW CNN architecture, output shape and backpropagation")
    model = ResNet18()
    logits = model(augment_batch(torch.rand(2, 3, 32, 32)))
    assert logits.shape == (2, 10)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    passed("Handmade ResNet-18 augmentation/forward/backward with finite gradients")
    vae = VAE(size=64, latent=16, base=8)
    x = torch.rand(2, 1, 64, 64)
    rec, mu, logvar = vae(x)
    assert rec.shape == x.shape and mu.shape == (2, 16)
    loss, _, _ = vae_loss(rec, x, mu, logvar, .001)
    loss.backward()
    assert vae.mu.weight.grad.abs().sum() > 0 and vae.logvar.weight.grad.abs().sum() > 0
    passed("VAE reparameterization propagates gradients into mean and log variance")
    unet = UNet(base=8)
    target = torch.randint(0, 4, (2, 64, 64))
    logits = unet(x)
    assert logits.shape == (2, 4, 64, 64)
    loss = segmentation_loss(logits, target)
    loss.backward()
    perfect = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 40 - 20
    assert segmentation_loss(perfect, target).item() < 1e-5
    assert dice_from_confusion(torch.eye(4, dtype=torch.long)) == [1., 1., 1., 1.]
    assert dice_from_confusion(torch.ones(4, 4, dtype=torch.long) - torch.eye(4, dtype=torch.long)) == [0.] * 4
    passed("UNet categorical output, one-hot loss and Dice edge cases")
    g, d = Generator(16, 8), Discriminator(8)
    fake = g(torch.randn(2, 16, 1, 1))
    assert fake.shape == (2, 1, 64, 64)
    g.zero_grad()
    d(fake.detach()).mean().backward()
    assert all(p.grad is None for p in g.parameters())
    d(fake).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    passed("GAN detach separates discriminator/generator gradient paths")
    save_json(ROOT / "verification/selftest.json", dict(passed=checks, synthetic=True,
              note="Correctness checks only, not evidence of classification/segmentation/generation quality."))


if __name__ == "__main__":
    main()
