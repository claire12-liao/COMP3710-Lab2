"""Part 3.2: handmade ResNet-18, GPU-resident CIFAR-10, AMP and live epoch demo."""
import argparse
import math
import os
import time
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from .models import ResNet18
from .common import (ROOT, torch, np, plt, seed_everything, device_for, environment, sync,
                     save_json, checkpoint, load_checkpoint, amp_context, scaler_for, curves)


def cached_cifar(root, device, smoke=False):
    datasets = [CIFAR10(str(root), train=train, download=True) for train in [True, False]]
    y = np.asarray(datasets[0].targets)
    train, val = train_test_split(np.arange(len(y)), test_size=.1, stratify=y, random_state=42)
    if smoke:
        train, val = train[:64], val[:32]
    tensors = {}
    for split, data, ids in [("train", datasets[0], train), ("val", datasets[0], val),
                             ("test", datasets[1], np.arange(32 if smoke else len(datasets[1])))]:
        x = torch.from_numpy(data.data[ids].copy()).permute(0, 3, 1, 2).to(device).float() / 255
        target = torch.tensor(np.asarray(data.targets)[ids], dtype=torch.long, device=device)
        tensors[split] = (x.contiguous(memory_format=torch.channels_last), target)
    return tensors, dict(train=train.tolist(), val=val.tolist())


def augment_batch(x, cutout=8):
    """Per-image random crop/flip/cutout, all on the tensor's device."""
    # DEMO P3-D: 提前把数据放到 GPU，并对一批图片一起做增强。
    # 减少逐张读取和传输的等待，让 GPU 把更多时间用于计算。
    batch, _, height, width = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    offsets = torch.randint(0, 9, (batch, 2), device=x.device)
    yy = offsets[:, 0, None, None] + torch.arange(height, device=x.device)[None, :, None]
    xx = offsets[:, 1, None, None] + torch.arange(width, device=x.device)[None, None, :]
    cropped = padded.permute(0, 2, 3, 1)[torch.arange(batch, device=x.device)[:, None, None], yy, xx]
    x = cropped.permute(0, 3, 1, 2)
    flipped = torch.rand(batch, 1, 1, 1, device=x.device) < .5
    x = torch.where(flipped, x.flip(-1), x)
    if cutout:
        centers = torch.randint(0, height, (batch, 2), device=x.device)
        rows = torch.arange(height, device=x.device)[None, :, None]
        cols = torch.arange(width, device=x.device)[None, None, :]
        hole = ((rows - centers[:, 0, None, None]).abs() < cutout // 2) & (
            (cols - centers[:, 1, None, None]).abs() < cutout // 2)
        x = x.masked_fill(hole[:, None], .5)
    return x


def normalize(x):
    mean = x.new_tensor([.4914, .4822, .4465])[None, :, None, None]
    std = x.new_tensor([.2470, .2435, .2616])[None, :, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def train_epoch(model, data, optimizer, scheduler, scaler, batch_size, device, amp=True):
    model.train()
    X, Y = data
    order = torch.randperm(len(Y), device=device)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for ids in order.split(batch_size):
        x = normalize(augment_batch(X[ids]))
        y = Y[ids]
        optimizer.zero_grad(set_to_none=True)
        # DEMO P3-E: autocast 为适合的运算使用较低精度，节省显存并加快 GPU 计算。
        # GradScaler 放大很小的梯度，降低 fp16 精度下梯度变成零的风险。
        with amp_context(device, amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach() * len(y)
        total_correct += (logits.detach().argmax(1) == y).sum()
    return float(total_loss.item() / len(Y)), float(total_correct.item() / len(Y))


@torch.no_grad()
def evaluate(model, data, batch_size, device, amp=True):
    model.eval()
    X, Y = data
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    predictions = []
    for start in range(0, len(Y), batch_size):
        x, y = X[start:start + batch_size], Y[start:start + batch_size]
        with amp_context(device, amp):
            logits = model(normalize(x))
            loss_sum += F.cross_entropy(logits, y, reduction="sum").float()
        pred = logits.argmax(1)
        correct += (pred == y).sum()
        predictions.append(pred.cpu())
    return loss_sum.item() / len(Y), correct.item() / len(Y), torch.cat(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "demo", "evaluate"], default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data", default=str(ROOT / "data/cifar10"))
    parser.add_argument("--output", default=str(ROOT / "runs/cifar"))
    parser.add_argument("--smoke", action="store_true", help="Tiny pipeline check, never a grade/accuracy result.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training with identical configuration.")
    parser.add_argument("--require-slurm", action="store_true", help="Fail if not running inside a Slurm GPU allocation.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    seed_everything()
    device = device_for(args.device)
    if args.require_slurm and (not os.environ.get("SLURM_JOB_ID") or device.type != "cuda" or args.smoke):
        raise RuntimeError("This demo requires a real Slurm GPU allocation and the full dataset.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    amp = not args.no_amp
    data, split = cached_cifar(args.data, device, args.smoke)
    save_json(out / "split_manifest.json", split)
    model = ResNet18().to(device=device, memory_format=torch.channels_last)
    best_path = out / "resnet18_best.pt"
    scaler = scaler_for(device, amp)

    if args.mode == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                                    weight_decay=5e-4, nesterov=True)
        steps = math.ceil(len(data["train"][1]) / args.batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * steps,
            pct_start=.2 if args.epochs * steps > 10 else .5, div_factor=25, final_div_factor=1000)
        history, best, first90, first94 = [], -1., None, None
        first_epoch, prior_elapsed = 1, 0.
        training_config = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               amp=amp, smoke=args.smoke)
        if args.resume:
            last = load_checkpoint(out / "resnet18_last.pt", device)
            if last["training_config"] != training_config:
                raise ValueError("Resume requires identical epochs, batch size, lr, AMP and dataset settings.")
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            scaler.load_state_dict(last["scaler"])
            import json
            history = json.loads((out / "history.json").read_text())
            first_epoch = last["epoch"] + 1
            prior_elapsed = history[-1]["elapsed_seconds"]
            best = load_checkpoint(best_path, device)["val_accuracy"]
            first90 = next((r["elapsed_seconds"] for r in history if r["val_accuracy"] > .9), None)
            first94 = next((r["elapsed_seconds"] for r in history if r["val_accuracy"] >= .94), None)
        sync(device)
        started = time.perf_counter()
        for epoch in range(first_epoch, args.epochs + 1):
            sync(device)
            epoch_start = time.perf_counter()
            train_loss, train_acc = train_epoch(model, data["train"], optimizer, scheduler,
                                                 scaler, args.batch_size, device, amp)
            val_loss, val_acc, _ = evaluate(model, data["val"], args.batch_size, device, amp)
            sync(device)
            elapsed = prior_elapsed + time.perf_counter() - started
            if val_acc > .9 and first90 is None:
                first90 = elapsed
            if val_acc >= .94 and first94 is None:
                first94 = elapsed
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, val_loss=val_loss,
                       val_accuracy=val_acc, elapsed_seconds=elapsed,
                       epoch_seconds=time.perf_counter() - epoch_start)
            history.append(row)
            if val_acc > best:
                best = val_acc
                checkpoint(best_path, dict(model=model.state_dict(), epoch=epoch, val_accuracy=best,
                                           elapsed_seconds=elapsed, smoke=args.smoke))
            # Recovery state is separate from the validation-selected demonstration checkpoint.
            checkpoint(out / "resnet18_last.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                       smoke=args.smoke, training_config=training_config))
            save_json(out / "history.json", history)
            print(f"ResNet18 {epoch}/{args.epochs} val={val_acc:.4f} elapsed={elapsed:.1f}s", flush=True)
        sync(device)
        total_training_seconds = prior_elapsed + time.perf_counter() - started
        curves(history, out / "accuracy.png", ["train_accuracy", "val_accuracy"], "CIFAR-10 accuracy")
        curves(history, out / "loss.png", ["train_loss", "val_loss"], "CIFAR-10 loss")
        timing = dict(total_training_wall_seconds=total_training_seconds,
                      resumed=args.resume,
                      time_to_90_validation=first90, time_to_94_validation=first94,
                      scope="All epochs+validation+checkpoint I/O; excludes data download/loading. Test reported once.")
        save_json(out / "training_timing.json", timing)
    saved = load_checkpoint(best_path, device)
    if bool(saved.get("smoke")) != args.smoke:
        raise ValueError("Smoke checkpoint cannot be presented as a full-data trained checkpoint.")
    model.load_state_dict(saved["model"])
    sync(device)
    test_start = time.perf_counter()
    _, accuracy, predictions = evaluate(model, data["test"], args.batch_size, device, amp)
    sync(device)
    result = dict(test_accuracy=accuracy, selected_epoch=saved["epoch"],
                  inference_seconds=time.perf_counter() - test_start,
                  train_samples=len(data["train"][1]), test_samples=len(data["test"][1]),
                  environment=environment(device), smoke=args.smoke, parameters=vars(args))
    if args.mode == "train":
        result.update(timing)
        result["test_above_90"] = accuracy > .9 and not args.smoke
        result["test_at_least_94"] = accuracy >= .94 and not args.smoke
        result["test_94_within_360_seconds"] = accuracy >= .94 and total_training_seconds <= 360 and not args.smoke and not args.resume
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data["test"][0][i].cpu().permute(1, 2, 0))
        ax.set_title(f"True {int(data['test'][1][i])} / Pred {int(predictions[i])}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=140)
    plt.close(fig)
    print("Held-out CIFAR-10 test accuracy:", accuracy, flush=True)
    if args.mode == "demo":
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
        sync(device)
        start = time.perf_counter()
        loss, acc = train_epoch(model, data["train"], optimizer, None, scaler, args.batch_size, device, amp)
        sync(device)
        result.update(live_epoch_seconds=time.perf_counter() - start, live_train_loss=loss,
                      live_train_accuracy=acc, live_epoch_samples=len(data["train"][1]),
                      full_epoch_completed=not args.smoke and len(data["train"][1]) == 45000)
        # DEMO P3-F: 现场模式会实际遍历完整训练集一轮，并完成参数更新。
        # 这一轮用于展示训练过程，不覆盖此前保存的最佳模型。
        save_json(out / "live_demo.json", result)
        print("LIVE EPOCH COMPLETE:", result["live_epoch_samples"], "samples", result["live_epoch_seconds"], "seconds")
    else:
        save_json(out / ("metrics.json" if args.mode == "train" else "evaluation.json"), result)


if __name__ == "__main__":
    main()
