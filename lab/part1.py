"""Part 1: Fourier reconstruction, spectra and honest CPU/GPU DFT timings."""
import argparse
import statistics
import time

from .common import ROOT, plt, np, torch, seed_everything, device_for, sync, environment, save_json


def square_wave(t, f0=1.0):
    return torch.sign(torch.sin(2 * torch.pi * f0 * t))


def square_wave_fourier(t, f0=1.0, count=50):
    # DEMO P1-A: 将 1、3、5…倍基频的正弦波相加来逼近方波。
    # 50 项对应 1 到 99 倍基频，频率越高的项权重越小。
    n = torch.arange(1, 2 * count, 2, device=t.device, dtype=t.dtype)[:, None]
    return (4 / torch.pi) * (torch.sin(2 * torch.pi * n * f0 * t[None, :]) / n).sum(0)


def naive_dft_numpy(x):
    size = len(x)
    out = np.zeros(size, dtype=np.complex128)
    for k in range(size):
        for n in range(size):
            out[k] += x[n] * np.exp(-2j * np.pi * k * n / size)
    return out


def naive_dft_torch(x, block=256):
    """Explicit DFT, O(N^2); block rows limit memory. No built-in FFT."""
    # DEMO P1-B: 一次生成一批频率的 DFT 系数，再用矩阵乘法计算各频率的分量。
    # 分块可减少内存占用；GPU 同时处理更多运算，但复杂度仍是 O(N²)。
    size = x.numel()
    n = torch.arange(size, dtype=x.dtype, device=x.device)[None, :]
    out = torch.empty(size, dtype=torch.complex128, device=x.device)
    for start in range(0, size, block):
        k = torch.arange(start, min(start + block, size), dtype=x.dtype, device=x.device)[:, None]
        matrix = torch.exp((-2j * torch.pi / size) * (k * n))
        out[start:start + len(k)] = matrix @ x.to(torch.complex128)
    return out


def naive_dft_gpu(x):
    if x.device.type != "cuda":
        raise ValueError("GPU DFT requires an actual CUDA tensor; CPU fallback is not GPU evidence.")
    return naive_dft_torch(x)


def timed(fn, device, repeats):
    fn()  # Warm-up: initialization is not mixed into steady-state timings.
    samples = []
    result = None
    for _ in range(repeats):
        # DEMO P1-C: GPU 提交任务后可能还没有算完，所以计时前后都等待 GPU 完成。
        # 这样测到的是实际计算耗时，而不只是发送任务的时间。
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=str(ROOT / "runs/part1"))
    args = parser.parse_args()
    if min(args.sizes) <= 198 or args.repeats < 1:
        parser.error("Use N > 198 to sample all 50 harmonics without aliasing, repeats >= 1.")
    seed_everything()
    from pathlib import Path
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    cpu = torch.device("cpu")
    t = torch.arange(2048, dtype=torch.float64) / 2048
    reference = square_wave(t)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes.flat[0].plot(t, reference, color="black")
    axes.flat[0].set_title("Original square wave")
    for ax, count in zip(list(axes.flat)[1:], [1, 3, 5, 20, 50]):
        ax.plot(t, square_wave_fourier(t, count=count), label=f"{count} odd harmonics")
        ax.plot(t, reference, "k--", alpha=.4)
        ax.set_title(f"Reconstruction: {count} harmonics")
        ax.legend()
    for ax in axes.flat:
        ax.set(xlabel="Time (s)", ylabel="Amplitude", ylim=(-1.5, 1.5))
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "reconstruction.png", dpi=140)
    plt.close(fig)

    records = []
    for size in args.sizes:
        time_axis = torch.arange(size, dtype=torch.float64) / size
        x = square_wave_fourier(time_axis)
        x_np = x.numpy()
        fft_reference = np.fft.fft(x_np)
        methods = [("NumPy naive DFT", lambda: naive_dft_numpy(x_np), cpu),
                   ("NumPy FFT", lambda: np.fft.fft(x_np), cpu),
                   ("PyTorch explicit DFT CPU", lambda: naive_dft_torch(x), cpu)]
        if device.type == "cuda":
            x_gpu = x.to(device)
            methods.append(("PyTorch explicit DFT GPU", lambda: naive_dft_gpu(x_gpu), device))
        for name, fn, dev in methods:
            value, median, samples = timed(fn, dev, args.repeats)
            value = value.cpu().numpy() if torch.is_tensor(value) else value
            error = float(np.max(np.abs(value - fft_reference)))
            agrees = bool(np.allclose(value, fft_reference, rtol=1e-8, atol=1e-7))
            if not agrees:
                raise AssertionError(f"DFT verification failed: {name}, N={size}, error={error}")
            records.append(dict(N=size, method=name, median_seconds=median,
                                samples_seconds=samples, max_abs_error=error, agrees=agrees))
            print(f"N={size:4d} | {name:28s} | {median:.6f} s | agrees={agrees}", flush=True)
        ranking = sorted([r for r in records if r["N"] == size], key=lambda r: r["median_seconds"])
        print("Fastest -> slowest:", " -> ".join(r["method"] for r in ranking), flush=True)

    size = max(args.sizes)
    x = square_wave_fourier(torch.arange(size, dtype=torch.float64) / size).numpy()
    freq = np.fft.rfftfreq(size, 1 / size)
    amp = np.abs(np.fft.rfft(x)) / size
    amp[1:-1 if size % 2 == 0 else None] *= 2
    odd = np.arange(1, 100, 2)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stem(freq, amp, basefmt=" ", label="Measured spectrum")
    ax.scatter(odd, 4 / (np.pi * odd), color="red", s=10, label="Expected 4/(pi*n)")
    ax.set(xlim=(0, 105), xlabel="Frequency (Hz)", ylabel="Single-sided amplitude",
           title="Recovering the 50 odd harmonics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "spectrum.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in sorted({r["method"] for r in records}):
        rows = sorted([r for r in records if r["method"] == method], key=lambda r: r["N"])
        ax.loglog([r["N"] for r in rows], [r["median_seconds"] for r in rows], "o-", label=method)
    ax.set(xlabel="Number of samples", ylabel="Median seconds", title="Measured DFT/FFT scaling")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "timings.png", dpi=140)
    plt.close(fig)
    report = dict(environment=environment(device), gpu_measured=device.type == "cuda", records=records,
                  timing_scope="Warm median; explicit DFT includes matrix creation; GPU excludes host transfers.",
                  parameters=vars(args))
    save_json(out / "metrics.json", report)
    print("Saved:", out, "| GPU measured:", report["gpu_measured"])


if __name__ == "__main__":
    main()
