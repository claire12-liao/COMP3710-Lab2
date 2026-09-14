"""Part 4: train/demo VAE, four-class UNet and DCGAN on the supplied OASIS PNG data."""
import argparse
import time
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .data import MRIDataset, pairs
from .models import VAE, UNet, Generator, Discriminator, init_gan
from .common import (ROOT, np, torch, plt, seed_everything, device_for, environment, sync,
                     save_json, checkpoint, load_checkpoint, amp_context, scaler_for, curves, image_grid)


def vae_loss(reconstruction, x, mu, logvar, beta):
    reconstruction_loss = F.mse_loss(reconstruction.float(), x.float())
    # DEMO P4-D: MSE 比较重建图与原图的像素差，越小表示重建越接近原图。
    # KL 鼓励潜在分布接近标准正态；beta 控制这项约束的强度。
    kl = -.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp()).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def segmentation_loss(logits, target):
    # DEMO P4-E: 把每个像素的类别变成四位 one-hot 标签，例如类别 2 对应 [0,0,1,0]。
    # 交叉熵判断类别是否正确，soft Dice 关注区域重叠；这是分类而不是预测灰度值。
    onehot = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
    log_prob = F.log_softmax(logits.float(), dim=1)
    ce = -(onehot * log_prob).sum(1).mean()
    prob = log_prob.exp()
    intersection = (prob * onehot).sum((0, 2, 3))
    denominator = (prob + onehot).sum((0, 2, 3))
    dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return .5 * ce + .5 * dice_loss


def dice_from_confusion(confusion):
    # DEMO P4-F: 分别计算每一类预测区域与真实区域的重叠程度，Dice 越接近 1 越好。
    # 不能只报整体像素准确率，因为大量背景像素可能掩盖脑组织分类错误。
    denominator = confusion.sum(0) + confusion.sum(1)
    scores = 2 * confusion.diag().double() / denominator.clamp_min(1)
    return [float(scores[i]) if denominator[i] > 0 else None for i in range(4)]


@torch.no_grad()
def evaluate_segmentation(model, loader, device, amp):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
    total_loss = 0.
    image_scores = [[] for _ in range(4)]
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        with amp_context(device, amp):
            logits = model(x)
            loss = segmentation_loss(logits, target)
        total_loss += loss.item() * len(x)
        prediction = logits.argmax(1)
        confusion += torch.bincount((target * 4 + prediction).flatten(), minlength=16).reshape(4, 4)
        for k in range(4):
            intersection = ((prediction == k) & (target == k)).sum((1, 2))
            denom = (prediction == k).sum((1, 2)) + (target == k).sum((1, 2))
            valid = denom > 0
            image_scores[k].extend((2 * intersection[valid] / denom[valid]).cpu().tolist())
    scores = dice_from_confusion(confusion)
    return dict(loss=total_loss / len(loader.dataset), dice_per_class=scores,
                mean_dice=float(np.mean([v for v in scores if v is not None])),
                mean_image_dice_per_class=[float(np.mean(x)) if x else None for x in image_scores],
                all_labels_above_0_9=all(v is not None and v > .9 for v in scores),
                confusion=confusion.cpu().tolist(), samples=len(loader.dataset),
                note="Global pixel-count DSC per class; image means exclude only jointly absent classes.")


@torch.no_grad()
def evaluate_vae(model, loader, device, beta):
    model.eval()
    sums = np.zeros(3)
    for x, _ in loader:
        x = x.to(device)
        reconstruction, mu, logvar = model(x)
        values = vae_loss(reconstruction, x, mu, logvar, beta)
        sums += np.array([v.item() for v in values]) * len(x)
    return dict(zip(["loss", "reconstruction_mse", "kl_mean"], (sums / len(loader.dataset)).tolist()))


@torch.no_grad()
def visualise_vae(model, dataset, loader, device, out):
    model.eval()
    chosen = np.linspace(0, len(dataset) - 1, min(8, len(dataset))).astype(int)
    x = torch.stack([dataset[int(i)][0] for i in chosen]).to(device)
    reconstructed, _, _ = model(x)
    image_grid(torch.cat([x, reconstructed]), out / "reconstructions.png",
               "Top: held-out MRI | Bottom: deterministic VAE reconstruction", cols=len(x))
    coords = torch.linspace(-2, 2, 12, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    z = torch.zeros(144, model.latent, device=device)
    z[:, 0], z[:, 1] = xx.flatten(), yy.flatten()
    # DEMO P4-G: 只改变潜在向量的两个维度，其他维度固定为 0，观察生成图怎样变化。
    # 这张网格图是高维空间的一张二维切片，不能代表整个潜在空间。
    generated = model.decode(z)
    image_grid(generated, out / "latent_manifold.png", "2D latent slice: z0,z1 in [-2,2], remaining z=0", cols=12)
    means = []
    for x, _ in loader:
        mu, _ = model.encode(x.to(device))
        means.append(mu.cpu())
    means = torch.cat(means).numpy()
    centered = means - means.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [item[3] for item in dataset.items]
    points = ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=8, cmap="viridis")
    fig.colorbar(points, ax=ax, label="MRI slice index")
    ax.set(xlabel="Latent PCA 1", ylabel="Latent PCA 2", title="Held-out encoded MRI means")
    fig.tight_layout()
    fig.savefig(out / "latent_projection.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(out / "latent_coordinates.npz", means=means, projection=projected)


@torch.no_grad()
def visualise_segmentation(model, dataset, device, out):
    model.eval()
    ids = np.linspace(0, len(dataset) - 1, min(6, len(dataset))).astype(int)
    x = torch.stack([dataset[int(i)][0] for i in ids]).to(device)
    target = torch.stack([dataset[int(i)][1] for i in ids])
    probabilities = model(x).float().softmax(1).cpu()
    prediction = probabilities.argmax(1)
    categorical = F.one_hot(prediction, 4).permute(0, 3, 1, 2)
    np.savez_compressed(out / "categorical_predictions.npz", probabilities=probabilities.numpy(),
                        prediction_one_hot=categorical.numpy().astype(np.uint8), targets=target.numpy(),
                        source_filenames=np.array([dataset.items[int(i)][0].name for i in ids]))
    fig, axes = plt.subplots(len(ids), 3, figsize=(9, len(ids) * 2.5), squeeze=False)
    for row in range(len(ids)):
        axes[row, 0].imshow(x[row, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(target[row], cmap="viridis", vmin=0, vmax=3)
        axes[row, 2].imshow(prediction[row], cmap="viridis", vmin=0, vmax=3)
        for ax, title in zip(axes[row], ["MRI", "Ground truth", "Prediction"]):
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "segmentations.png", dpi=140)
    plt.close(fig)


def supervised(args, device, out):
    config = dict(size=args.size, latent=args.latent, base=args.base)
    ckpt_path = out / f"{args.task}_best.pt"
    saved = load_checkpoint(ckpt_path, device) if args.mode == "demo" else None
    if saved is not None:
        config = saved["config"]
        if saved.get("limited_data", False) and args.limit == 0:
            raise ValueError("This checkpoint used a limited training subset; set --limit explicitly for a smoke demo.")
    model = (VAE(**config) if args.task == "vae" else UNet(base=config["base"])).to(device)
    def dataset(split, augment=False):
        return MRIDataset(args.data, split, size=config["size"], limit=args.limit, augment=augment)
    if args.mode == "train":
        train_set, val_set = dataset("train", args.task == "unet"), dataset("validate")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = scaler_for(device, args.task == "unet")
        best, history, first_epoch, best_mean = float("inf"), [], 1, -1.
        if args.resume:
            last = load_checkpoint(out / f"{args.task}_last.pt", device)
            if last["config"] != config or bool(last["limited_data"]) != (args.limit > 0):
                raise ValueError("Resume requires the same architecture and full/subset setting.")
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            scaler.load_state_dict(last["scaler"])
            import json
            history = json.loads((out / "history.json").read_text())
            first_epoch = last["epoch"] + 1
            previous = load_checkpoint(ckpt_path, device)["metric"]
            best = previous["loss"] if args.task == "vae" else -min(v for v in previous["dice_per_class"] if v is not None)
            best_mean = previous.get("mean_dice", -1.)
        sync(device)
        started = time.perf_counter()
        for epoch in range(first_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.
            beta = args.beta * min(1., epoch / 10.)
            for x, target in train_loader:
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, args.task == "unet"):
                    if args.task == "vae":
                        reconstruction, mu, logvar = model(x)
                        loss, _, _ = vae_loss(reconstruction, x, mu, logvar, beta)
                    else:
                        loss = segmentation_loss(model(x), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; training stopped to protect checkpoint integrity.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(x)
            # Fixed beta for validation makes scores comparable during training warm-up.
            metric = (evaluate_vae(model, val_loader, device, args.beta) if args.task == "vae"
                      else evaluate_segmentation(model, val_loader, device, True))
            score = metric["loss"] if args.task == "vae" else -min(v for v in metric["dice_per_class"] if v is not None)
            row = dict(epoch=epoch, train_loss=total_loss / len(train_set), val_loss=metric["loss"],
                       val_metric=metric, elapsed_seconds=time.perf_counter() - started)
            if args.task == "unet":
                row["val_min_class_dice"] = -score
            history.append(row)
            if score < best or (args.task == "unet" and score == best and metric["mean_dice"] > best_mean):
                best = score
                best_mean = metric.get("mean_dice", -1.)
                checkpoint(ckpt_path, dict(model=model.state_dict(), config=config, epoch=epoch,
                           metric=metric, beta=args.beta, limited_data=args.limit > 0))
            scheduler.step()
            checkpoint(out / f"{args.task}_last.pt", dict(model=model.state_dict(), config=config,
                       epoch=epoch, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), beta=args.beta, limited_data=args.limit > 0))
            save_json(out / "history.json", history)
            print(f"{args.task} {epoch}/{args.epochs} train={row['train_loss']:.5f} val={metric}", flush=True)
        curves(history, out / "losses.png", ["train_loss", "val_loss"], args.task.upper() + " learning curves")
        saved = load_checkpoint(ckpt_path, device)
    model.load_state_dict(saved["model"])
    test_set = dataset("test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    # DEMO P4-H: 用验证集决定保存哪一个模型，再用独立测试集报告最终表现。
    # 避免用测试成绩反复挑模型，否则最终分数会过于乐观。
    sync(device)
    start = time.perf_counter()
    if args.task == "vae":
        metric = evaluate_vae(model, test_loader, device, saved["beta"])
        visualise_vae(model, test_set, test_loader, device, out)
    else:
        metric = evaluate_segmentation(model, test_loader, device, True)
        visualise_segmentation(model, test_set, device, out)
    sync(device)
    metric.update(selected_epoch=saved["epoch"], validation_metric=saved["metric"],
                  environment=environment(device), limited_data=saved["limited_data"],
                  inference_and_plots_seconds=time.perf_counter() - start, parameters=vars(args),
                  config=config, mode=args.mode)
    save_json(out / ("live_demo.json" if args.mode == "demo" else "metrics.json"), metric)
    print("HELD-OUT TEST:", metric, flush=True)


def gan(args, device, out):
    if args.size != 64:
        raise ValueError("This DCGAN is designed for 64x64 MRI. Use --size 64.")
    ckpt_path = out / "gan_final.pt"
    saved = load_checkpoint(ckpt_path, device) if args.mode == "demo" else None
    latent = saved["latent"] if saved else args.latent
    base = saved["base"] if saved else args.base
    generator, discriminator = Generator(latent, base).to(device), Discriminator(base).to(device)
    train_set = MRIDataset(args.data, "train", size=64, limit=args.limit)
    if len(train_set) < 2:
        raise ValueError("GAN requires at least two training samples.")
    fixed = torch.randn(32, latent, 1, 1, device=device)
    if args.mode == "train":
        generator.apply(init_gan)
        discriminator.apply(init_gan)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if len(loader) == 0:
            raise ValueError("Set batch-size <= training samples.")
        opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(.5, .999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.5, .999))
        history, first_epoch = [], 1
        if args.resume:
            last = load_checkpoint(ckpt_path, device)
            if last["latent"] != latent or last["base"] != base or bool(last["limited_data"]) != (args.limit > 0):
                raise ValueError("Resume requires the same GAN configuration and full/subset setting.")
            generator.load_state_dict(last["generator"])
            discriminator.load_state_dict(last["discriminator"])
            opt_g.load_state_dict(last["optimizer_g"])
            opt_d.load_state_dict(last["optimizer_d"])
            fixed = last["fixed_noise"].to(device)
            import json
            history = json.loads((out / "history.json").read_text())
            first_epoch = last["epoch"] + 1
        for epoch in range(first_epoch, args.epochs + 1):
            generator.train()
            discriminator.train()
            g_sum, d_sum, count = 0., 0., 0
            for x, _ in loader:
                real = x.to(device) * 2 - 1
                z = torch.randn(len(x), latent, 1, 1, device=device)
                fake = generator(z)
                # DEMO P4-I: 训练判别器 D 时，用 detach 切断假图与生成器的梯度连接。
                # 这一步只更新 D；训练生成器 G 时保留连接，让“假图被识破”的误差传回 G。
                discriminator.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                real_logits, fake_logits = discriminator(real), discriminator(fake.detach())
                loss_d = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, .9)) + \
                         F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                loss_d.backward()
                opt_d.step()
                discriminator.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                fake_logits = discriminator(fake)
                loss_g = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                loss_g.backward()
                opt_g.step()
                g_sum += loss_g.item() * len(x)
                d_sum += loss_d.item() * len(x)
                count += len(x)
            row = dict(epoch=epoch, generator_loss=g_sum / count, discriminator_loss=d_sum / count)
            history.append(row)
            generator.eval()
            with torch.no_grad():
                image_grid((generator(fixed) + 1) / 2, out / f"samples_epoch_{epoch:03d}.png",
                           f"GAN epoch {epoch}: fixed noise (training progress)")
            checkpoint(ckpt_path, dict(generator=generator.state_dict(), discriminator=discriminator.state_dict(),
                       optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(),
                       latent=latent, base=base, epoch=epoch, limited_data=args.limit > 0, fixed_noise=fixed))
            save_json(out / "history.json", history)
            print("GAN", row, flush=True)
        curves(history, out / "losses.png", ["generator_loss", "discriminator_loss"], "GAN training losses")
        saved = load_checkpoint(ckpt_path, device)
    if saved.get("limited_data", False) and args.limit == 0:
        raise ValueError("Limited-data checkpoint: use --limit explicitly for a smoke demonstration.")
    generator.load_state_dict(saved["generator"])
    generator.eval()
    with torch.no_grad():
        samples = (generator(torch.randn(32, latent, 1, 1, device=device)) + 1) / 2
        image_grid(samples, out / "generated.png", "New MRI samples from fresh random noise")
        left, right = torch.randn(2, latent, 1, 1, device=device)
        weights = torch.linspace(0, 1, 12, device=device)[:, None, None, None]
        images = (generator((1 - weights) * left + weights * right) + 1) / 2
        image_grid(images, out / "interpolation.png", "Noise interpolation: continuity and diversity", cols=12)
        # Diagnostics, not a substitute for tutor judgment of realism or memorization.
        generated_small = F.interpolate(samples, size=(16, 16), mode="area").flatten(1).cpu()
        reference = F.interpolate(train_set.images.float() / 255, size=(16, 16), mode="area").flatten(1)
        distances = torch.cdist(generated_small, reference) / 16
        nearest, nearest_ids = distances.min(1)
        references = torch.stack([train_set[int(i)][0] for i in nearest_ids[:8]])
        image_grid(torch.cat([samples[:8].cpu(), references]), out / "nearest_training_examples.png",
                   "Top: generated | Bottom: nearest training MRI at 16x16 (diagnostic only)", cols=8)
        metric = dict(environment=environment(device), selected_epoch=saved["epoch"],
                      actual_model_config=dict(latent=latent, base=base, size=64),
                      limited_data=saved["limited_data"], generated_count=len(samples),
                      pairwise_thumbnail_rmse_mean=float(torch.pdist(generated_small).mean() / 16),
                      nearest_training_thumbnail_rmse=nearest.tolist(),
                      realism_verified=False, note="Visual tutor assessment still required; no automatic realism pass.",
                      parameters=vars(args))
        save_json(out / ("live_demo.json" if args.mode == "demo" else "metrics.json"), metric)
    print("GAN samples saved. Inspect realism, diversity and nearest-training comparisons:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["vae", "unet", "gan"])
    parser.add_argument("--mode", choices=["train", "demo"], default="train")
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--base", type=int, default=None)
    parser.add_argument("--latent", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=.001)
    parser.add_argument("--limit", type=int, default=0, help="Subset testing only; 0 uses full split.")
    parser.add_argument("--resume", action="store_true", help="Resume a saved run, using the same training configuration.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    defaults = dict(vae=(40, 64, 64, 32, 16, .001), unet=(80, 8, 256, 32, 16, .001),
                    gan=(100, 64, 64, 64, 100, .0002))[args.task]
    for key, value in zip(["epochs", "batch_size", "size", "base", "latent", "lr"], defaults):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.latent < 2 or args.epochs < 1 or args.batch_size < 1:
        parser.error("latent >=2, epochs >=1 and batch-size >=1 are required")
    seed_everything()
    device = device_for(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # Quick mandatory subject check before training, even if a full pixel audit was not run.
    ids = {s: {r[2] for r in pairs(args.data, s)} for s in ["train", "validate", "test"]}
    if ids["train"] & ids["validate"] or ids["train"] & ids["test"] or ids["validate"] & ids["test"]:
        raise ValueError("MRI subject leakage detected")
    out = Path(args.output or ROOT / "runs" / args.task)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_config.json", dict(parameters=vars(args), environment=environment(device)))
    if args.task == "gan":
        gan(args, device, out)
    else:
        supervised(args, device, out)


if __name__ == "__main__":
    main()
