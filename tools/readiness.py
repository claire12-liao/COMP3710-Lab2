"""Generate a factual completion dashboard from this user's runs, never from smoke checks."""
from pathlib import Path
import json
import html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def read(relative):
    path = RUNS / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    p1, rf, cnn = read("part1/metrics.json"), read("faces/rf_metrics.json"), read("faces/cnn_metrics.json")
    cifar, live = read("cifar/metrics.json"), read("cifar/live_demo.json")
    vae, unet, gan = read("vae/metrics.json"), read("unet/metrics.json"), read("gan/metrics.json")
    rows = [
        ("Part 1: wave / spectrum / sizes", bool(p1), "runs/part1/*.png, metrics.json"),
        ("Part 1: actual GPU DFT timing", p1.get("gpu_measured", False), "A CPU run is insufficient for GPU timing."),
        ("Part 2: PCA + Random Forest", bool(rf), str(rf.get("test_accuracy", "Not run"))),
        ("Part 3.1: LFW CNN", bool(cnn), str(cnn.get("test_accuracy", "Not run"))),
        ("Part 3.2: test accuracy > 90%", cifar.get("test_above_90", False), str(cifar.get("test_accuracy", "Not run"))),
        ("Part 3.2: test accuracy >= 94%, <=360s", cifar.get("test_94_within_360_seconds", False),
         "Compare hardware fairly; validation accuracy is not test accuracy."),
        ("Part 3.2: GPU live full training epoch", live.get("full_epoch_completed", False) and
         bool(live.get("environment", {}).get("slurm_job_id")) and live.get("environment", {}).get("device") == "cuda",
         "Tutor must verify this was on Rangpur, during your actual demonstration."),
        ("Part 4 VAE: full-data run recorded", bool(vae) and not vae.get("limited_data", True),
         "Manifold/reconstruction quality still needs inspection."),
        ("Part 4 UNet: all labels DSC > 0.9", unet.get("all_labels_above_0_9", False) and not unet.get("limited_data", True),
         str(unet.get("dice_per_class", "Not run"))),
        ("Part 4 UNet: live test inference recorded", bool(read("unet/live_demo.json")) and
         not read("unet/live_demo.json").get("limited_data", True), "Run again live for the tutor."),
        ("Part 4 GAN: full-data run recorded", bool(gan) and not gan.get("limited_data", True),
         "Realism, diversity and mode collapse require human judgment."),
        ("Git advanced course", None, "Complete personally; save evidence from this semester's course."),
        ("Own GitHub account and meaningful commits", None, "Verify repository, README and actual commit history."),
        ("Understanding / AI evidence / live presentation", None, "Practise docs/DEMO_GUIDE.md; add your actual changes and conversation evidence.")]
    RUNS.mkdir(exist_ok=True)
    body = "".join("<tr><td>" + html.escape(name) + "</td><td>" +
                   ("Evidence found" if status is True else "Pending" if status is False else "Check manually") +
                   "</td><td>" + html.escape(detail) + "</td></tr>" for name, status, detail in rows)
    page = """<!doctype html><meta charset='utf-8'><title>COMP3710 readiness</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#172331}
table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}
th{background:#eef4f8}h1{color:#163f60}</style><h1>COMP3710 Lab 2 — Evidence checklist</h1>
<p>本表检查 runs/ 中的真实结果，不使用 verification/ 中的小样本测试；不是自动评分或满分保证。</p>
<table><tr><th>Criterion</th><th>Status</th><th>Evidence / remaining action</th></tr>""" + body + "</table>"
    (RUNS / "READINESS.html").write_text(page, encoding="utf-8")
    for name, status, detail in rows:
        print(("FOUND" if status else "MANUAL" if status is None else "PENDING"), "|", name, "|", detail)
    print("Saved", RUNS / "READINESS.html")


if __name__ == "__main__":
    main()
