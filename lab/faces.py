"""Parts 2 and 3.1: shared LFW splits, NumPy PCA+RF, and the required two-convolution CNN."""
import argparse
from pathlib import Path
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .common import (ROOT, np, torch, plt, seed_everything, device_for, environment,
                     save_json, checkpoint, load_checkpoint, curves)
from .models import FaceCNN


def load_faces(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / "lfw_shared_split.npz"
    if not file.exists():
        try:
            data = fetch_lfw_people(min_faces_per_person=70, resize=.4, data_home=str(cache))
        except Exception as exc:
            raise RuntimeError("LFW download/load failed. Use a network-enabled Colab/course environment "
                               "or copy the original sklearn LFW cache into --data. Do not substitute a different dataset. "
                               f"Original error: {exc}") from exc
        images = data.images.astype(np.float32)
        if images.max() > 1:
            images /= 255.
        indices = np.arange(len(images))
        # DEMO P2-A: 训练集学习参数，验证集选模型，测试集用于最终评估。
        # PCA 和 CNN 共用相同划分，比较时不会因测试图片不同而产生偏差。
        train_val, test = train_test_split(indices, test_size=.25, random_state=42, stratify=data.target)
        train, val = train_test_split(train_val, test_size=.2, random_state=42,
                                     stratify=data.target[train_val])
        np.savez_compressed(file, images=images, y=data.target, names=data.target_names,
                            train=train, val=val, test=test)
    with np.load(file, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def report_predictions(y, pred, names, out, prefix):
    labels = np.arange(len(names))
    report = classification_report(y, pred, labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    save_json(out / f"{prefix}_report.json", report)
    text = classification_report(y, pred, labels=labels, target_names=names, zero_division=0)
    (out / f"{prefix}_report.txt").write_text(text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=labels, display_labels=names,
                                            ax=ax, xticks_rotation=70, colorbar=False)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_confusion.png", dpi=140)
    plt.close(fig)
    print(text)
    return float(accuracy_score(y, pred))


def pca_rf(data, out):
    images, labels, names = data["images"], data["y"], data["names"]
    train, val, test = data["train"], data["val"], data["test"]
    X = images.reshape(len(images), -1).astype(np.float64)
    # DEMO P2-B: 先减去训练集平均脸，再用 SVD 找到变化最明显的方向作为特征脸。
    # 验证和测试图片也使用这组均值与方向，不能提前让测试数据参与学习。
    mean = X[train].mean(0)
    centered = X[train] - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(150, len(train) - 1, X.shape[1])
    components = vt[:count]
    projected = (X - mean) @ components.T
    variance = singular ** 2 / (len(train) - 1)
    cumulative = np.cumsum(variance / variance.sum())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, count + 1), cumulative[:count])
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance", title="PCA compactness")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "compactness.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(8, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(components[i].reshape(images.shape[1:]), cmap="gray")
        ax.set_title(f"Eigenface {i + 1}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "eigenfaces.png", dpi=140)
    plt.close(fig)
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=count,
                                random_state=42, n_jobs=4)
    rf.fit(projected[train], labels[train])
    val_acc = float(rf.score(projected[val], labels[val]))
    prediction = rf.predict(projected[test])
    test_acc = report_predictions(labels[test], prediction, names, out, "rf")
    # Save the PCA basis; RF can be refitted from this fixed split with --mode rf.
    np.savez_compressed(out / "pca_basis.npz", mean=mean, components=components,
                        singular_values=singular, train=train, val=val, test=test)
    result = dict(validation_accuracy=val_acc, test_accuracy=test_acc, components=count,
                  retained_variance=float(cumulative[count - 1]), train_samples=len(train),
                  validation_samples=len(val), test_samples=len(test), seed=42)
    save_json(out / "rf_metrics.json", result)
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, targets, predictions = 0., [], []
    for x, y in loader:
        logits = model(x.to(device))
        total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        targets.extend(y.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(targets), float(accuracy_score(targets, predictions)), targets, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "rf", "train", "demo"], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data", default=str(ROOT / "data/lfw"))
    parser.add_argument("--output", default=str(ROOT / "runs/faces"))
    args = parser.parse_args()
    seed_everything()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    data = load_faces(args.data)
    save_json(out / "split_manifest.json", {k: data[k].tolist() for k in ["train", "val", "test", "names"]})
    if args.mode in ["all", "rf"]:
        pca_rf(data, out)
    if args.mode == "rf":
        return
    # DEMO P3-C: 卷积输入按 NCHW 排列：图片数量、颜色通道、高度、宽度。
    # 灰度人脸只有一个通道，补上这个维度后才能传给二维卷积层。
    X = torch.from_numpy(data["images"][:, None])
    y = torch.from_numpy(data["y"].astype(np.int64))
    loaders = {split: DataLoader(TensorDataset(X[data[split]], y[data[split]]),
                                batch_size=args.batch_size, shuffle=split == "train")
               for split in ["train", "val", "test"]}
    model = FaceCNN(len(data["names"])).to(device)
    model_path = out / "face_cnn_best.pt"
    if args.mode != "demo":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history, best = [], -1.
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.
            for x, target in loaders["train"]:
                x, target = x.to(device), target.to(device)
                flip = torch.rand(len(x), device=device) < .5
                x[flip] = x[flip].flip(-1)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), target)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
            val_loss, val_acc, _, _ = evaluate(model, loaders["val"], device)
            scheduler.step()
            history.append(dict(epoch=epoch, train_loss=total / len(data["train"]),
                                val_loss=val_loss, val_accuracy=val_acc))
            if val_acc > best:
                best = val_acc
                checkpoint(model_path, dict(model=model.state_dict(), classes=len(data["names"]),
                                           epoch=epoch, val_accuracy=best))
            save_json(out / "history.json", history)
            print(f"FaceCNN epoch {epoch}/{args.epochs}: validation accuracy={val_acc:.4f}", flush=True)
        curves(history, out / "cnn_losses.png", ["train_loss", "val_loss"], "LFW CNN learning curves")
    saved = load_checkpoint(model_path, device)
    model.load_state_dict(saved["model"])
    _, test_acc, truth, pred = evaluate(model, loaders["test"], device)
    report_predictions(truth, pred, data["names"], out, "cnn")
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data["images"][data["test"][i]], cmap="gray")
        ax.set_title(f"True: {data['names'][truth[i]]}\nPred: {data['names'][pred[i]]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "cnn_predictions.png", dpi=140)
    plt.close(fig)
    result = dict(test_accuracy=test_acc, best_validation_accuracy=saved["val_accuracy"],
                  selected_epoch=saved["epoch"], environment=environment(device), parameters=vars(args))
    save_json(out / "cnn_metrics.json", result)
    rf_path = out / "rf_metrics.json"
    if rf_path.exists():
        import json
        rf = json.loads(rf_path.read_text())
        save_json(out / "comparison.json", dict(rf_test_accuracy=rf["test_accuracy"],
                                                cnn_test_accuracy=test_acc,
                                                cnn_outperformed_rf=test_acc > rf["test_accuracy"]))
    print("CNN held-out test accuracy:", test_acc)


if __name__ == "__main__":
    main()
