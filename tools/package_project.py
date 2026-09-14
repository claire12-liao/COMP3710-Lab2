"""Create downloadable notebook and complete ZIP; exclude large untrained checkpoints/caches."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "COMP3710_Lab2_Colab.ipynb", OUTPUT / "COMP3710_Lab2_Colab.ipynb")
    selected = []
    for file in sorted(ROOT.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(ROOT)
        if "__pycache__" in rel.parts or file.suffix in {".pt", ".pth", ".pyc", ".tmp"}:
            continue
        if rel.parts[0] == "data" and (len(rel.parts) < 2 or rel.parts[1] != "keras_png_slices_data"):
            continue
        if rel.parts[0] in {"runs", ".venv"}:
            continue
        selected.append(file)
    destination = OUTPUT / "COMP3710_Lab2_Complete.zip"
    hashes = {}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in selected:
            rel = file.relative_to(ROOT)
            archive.write(file, "COMP3710_Lab2/" + rel.as_posix())
            if rel.parts[0] != "data":
                hashes[rel.as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        archive.writestr("COMP3710_Lab2/FILE_CHECKSUMS.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert sum(n.endswith(".png") and "/data/" in n for n in names) == 22656
        assert all(not n.endswith((".pt", ".pth", ".pyc")) for n in names)
        assert "COMP3710_Lab2/COMP3710_Lab2_Colab.ipynb" in names
        assert "COMP3710_Lab2/START_HERE.html" in names
    print("ZIP integrity and complete MRI file count verified.")
    print("Files:", len(selected), "| bytes:", destination.stat().st_size)
    print(destination)
    print(OUTPUT / "COMP3710_Lab2_Colab.ipynb")


if __name__ == "__main__":
    main()
