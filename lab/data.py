"""Dataset preparation with original MRI splits, exact pairing and subject checks."""
from pathlib import Path
import argparse
import re
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from .common import ROOT, np, torch, save_json


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP member: " + info.filename)
        z.extractall(destination)


def resolve_mri(root):
    root = Path(root)
    for candidate in [root, root / "keras_png_slices_data"]:
        if (candidate / "keras_png_slices_train").is_dir():
            return candidate
    raise FileNotFoundError(f"MRI folders not found under {root}. Run: python -m lab.data")


def extract_mri_archive(archive, destination):
    """Accept the original data ZIP or our complete-project ZIP; extract MRI PNGs only."""
    destination = Path(destination).resolve()
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if "keras_png_slices_data" not in parts or not info.filename.endswith(".png"):
                continue
            relative = Path(*parts[parts.index("keras_png_slices_data"):])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe MRI ZIP member")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            count += 1
    if count == 0:
        raise ValueError("No keras_png_slices_data/*.png found in this ZIP.")
    return resolve_mri(destination)


def pairs(root, split):
    root = resolve_mri(root)
    items = []
    for p in sorted((root / f"keras_png_slices_{split}").glob("*.png")):
        match = re.fullmatch(r"case_(\d+)_slice_(\d+)\.nii\.png", p.name)
        if match is None:
            raise ValueError("Unexpected MRI filename: " + p.name)
        subject, slice_id = match.groups()
        mask = root / f"keras_png_slices_seg_{split}" / f"seg_{subject}_slice_{slice_id}.nii.png"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        items.append((p, mask, subject, int(slice_id)))
    if not items:
        raise ValueError("No MRI pairs found: " + split)
    actual_masks = set((root / f"keras_png_slices_seg_{split}").glob("*.png"))
    if actual_masks != {r[1] for r in items}:
        raise ValueError("Extra or missing masks: " + split)
    return items


def audit_mri(root):
    report, subjects = {}, {}
    for split in ["train", "validate", "test"]:
        items = pairs(root, split)
        subjects[split] = {r[2] for r in items}
        values = set()
        shapes = set()
        for image, mask, _, _ in items:
            with Image.open(image) as im, Image.open(mask) as seg:
                if im.size != seg.size:
                    raise ValueError("Image/mask shape mismatch: " + str(image))
                shapes.add(im.size)
                values.update(np.unique(np.asarray(seg)).tolist())
        if values != {0, 85, 170, 255}:
            raise ValueError(f"Unexpected mask encoding {values}. Do not silently guess labels.")
        report[split] = dict(images=len(items), subjects=len(subjects[split]),
                             shapes=sorted(shapes), mask_values=sorted(values),
                             subject_ids=sorted(subjects[split]))
    for a, b in [("train", "validate"), ("train", "test"), ("validate", "test")]:
        if subjects[a] & subjects[b]:
            raise ValueError(f"Subject leakage: {a} / {b}")
    report["subject_disjoint"] = True
    report["label_mapping"] = {"0": 0, "85": 1, "170": 2, "255": 3}
    return report


class MRIDataset(Dataset):
    def __init__(self, root, split, size=64, limit=0, augment=False):
        self.items = pairs(root, split)
        if limit:
            # Deterministic, spread across subjects instead of only taking the first patient.
            ids = np.linspace(0, len(self.items) - 1, min(limit, len(self.items))).astype(int)
            self.items = [self.items[i] for i in ids]
        self.augment = augment
        images, masks = [], []
        for p, mask, _, _ in self.items:
            with Image.open(p) as image, Image.open(mask) as seg:
                images.append(np.array(image.convert("L").resize((size, size), Image.Resampling.BILINEAR)))
                raw = np.array(seg.convert("L").resize((size, size), Image.Resampling.NEAREST))
                if not np.isin(raw, [0, 85, 170, 255]).all():
                    raise ValueError("Unknown label gray value: " + str(mask))
                # DEMO P4-C: 标签缩放使用最近邻，避免插值产生原本不存在的类别值。
                # 把 0、85、170、255 转成 0、1、2、3，供四分类损失使用。
                masks.append(raw // 85)
        self.images = torch.from_numpy(np.stack(images))[:, None]
        self.masks = torch.from_numpy(np.stack(masks))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image = self.images[index].float() / 255
        mask = self.masks[index].long()
        if self.augment and torch.rand(()) < .5:
            image, mask = image.flip(-1), mask.flip(-1)
        return image, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=str(ROOT / "data/keras_png_slices_data.zip"))
    parser.add_argument("--root", default=str(ROOT / "data"))
    args = parser.parse_args()
    try:
        root = resolve_mri(args.root)
    except FileNotFoundError:
        extract_mri_archive(args.zip, args.root)
        root = resolve_mri(args.root)
    report = audit_mri(root)
    save_json(ROOT / "runs/data_audit.json", report)
    for split in ["train", "validate", "test"]:
        print(split, report[split]["images"], "images;", report[split]["subjects"], "subjects")
    print("Subject split audit passed. Root:", root)


if __name__ == "__main__":
    main()
