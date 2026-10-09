"""Evaluate three released DeepfakeBench detectors on author-preprocessed UADFV.
All project execution, weights, and dataset processing happen on a GitHub cloud runner.
"""
import os, sys, json, time, hashlib, platform, csv, random, warnings, runpy
from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import requests, gdown, cv2, numpy as np, torch, yaml
from PIL import Image

WORK = Path.cwd().resolve()
ROOT = WORK / "DeepfakeBench"
OUT = WORK / "results"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "datasets/rgb/UADFV_cloud"
DATA.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(1024)
np.random.seed(1024)
random.seed(1024)
warnings.filterwarnings("ignore", message="dropout2d: Received a 2-D input")

# Import only evaluated modules. Network architectures and forward methods remain original.
(ROOT / "training/detectors/__init__.py").write_text(
    "from metrics.registry import DETECTOR\nfrom .meso4_detector import Meso4Detector\n"
    "from .meso4Inception_detector import Meso4InceptionDetector\nfrom .xception_detector import XceptionDetector\n")
(ROOT / "training/networks/__init__.py").write_text(
    "from metrics.registry import BACKBONE\nfrom .mesonet import Meso4, MesoInception4\nfrom .xception import Xception\n")
(ROOT / "training/loss/__init__.py").write_text(
    "from metrics.registry import LOSSFUNC\nfrom .cross_entropy_loss import CrossEntropyLoss\n")
(ROOT / "training/dataset/__init__.py").write_text(
    "from .abstract_dataset import DeepfakeAbstractBaseDataset\n")
test_path = ROOT / "training/test.py"
original_test = test_path.read_text()
remove_lines = [
    "from dataset.ff_blend import FFBlendDataset",
    "from dataset.fwa_blend import FWABlendDataset",
    "from dataset.pair_dataset import pairDataset",
    "from trainer.trainer import Trainer",
]
test_path.write_text("\n".join(line for line in original_test.splitlines() if line not in remove_lines) + "\n")

sys.path.insert(0, str(ROOT / "training"))
from detectors import DETECTOR
from detectors.xception_detector import XceptionDetector
from networks import BACKBONE
# The complete trained detector checkpoint overwrites every parameter strictly.
# Skip only the redundant ImageNet backbone initialization required by the training constructor.
def inference_backbone(self, config):
    return BACKBONE[config["backbone_name"]](config["backbone_config"])
XceptionDetector.build_backbone = inference_backbone

report = {
    "run_location": "GitHub-hosted Ubuntu 22.04 standard cloud runner (CPU)",
    "python": sys.version.split()[0], "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "upstream_commit": "f188b1c105465e2e5377eb536a95022ae0e4522d",
    "dataset": "UADFV", "dataset_source": "Author-preprocessed RGB archive from the DeepfakeBench README",
    "data_url": "https://drive.google.com/file/d/1kEMijoDJKaIMKjJFK7YJVA71z3Femd0y/view",
    "scope": "Released pretrained detectors evaluated on all available UADFV videos, up to 32 preprocessed face frames per video; no training",
    "resolution": [256, 256], "normalization_mean": [0.5]*3, "normalization_std": [0.5]*3,
    "seed": 1024, "batch_size": 16, "cpu_threads": 4,
    "adaptations": [
        "Select only evaluated detectors, networks, losses, and dataset imports",
        "Remove unused training-only imports from test.py",
        "Skip redundant ImageNet backbone initialization before strict complete Xception checkpoint load",
        "Generate cloud paths from the author's archive; prefix video folders by class for unique video-level aggregation"
    ],
    "evaluation_functions": "Original training/test.py prepare_testing_data and test_epoch; original DeepfakeAbstractBaseDataset and metrics/utils.py get_test_metrics",
    "models": {}
}
(OUT / "run_manifest.json").write_text(json.dumps(report, indent=2))
archive = ROOT / "datasets/UADFV.zip"
print("DOWNLOADING AUTHOR-PREPROCESSED UADFV", flush=True)
gdown.download(id="1kEMijoDJKaIMKjJFK7YJVA71z3Femd0y", output=str(archive), quiet=False)
assert archive.exists(), "UADFV archive download failed"
report["archive_bytes"] = archive.stat().st_size
digest = hashlib.sha256()
with archive.open("rb") as f:
    for chunk in iter(lambda: f.read(1024*1024), b""):
        digest.update(chunk)
report["archive_sha256"] = digest.hexdigest()

import zipfile
groups = defaultdict(list)
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    (OUT / "archive_structure.txt").write_text("\n".join(names[:150]))
    for name in names:
        parts = PurePosixPath(name).parts
        lower = [p.lower() for p in parts]
        if "frames" not in lower or not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if "real" in lower:
            label = "UADFV_Real"
        elif "fake" in lower:
            label = "UADFV_Fake"
        else:
            raise ValueError("Unrecognized frame label directory: " + name)
        video_id = parts[-2]
        groups[(label, video_id)].append(name)
    assert groups, "No face frames found in author archive"
    manifest = {"UADFV": {label: {"test": {}} for label in ["UADFV_Real", "UADFV_Fake"]}}
    audit = []
    for (label, video_id), source_names in sorted(groups.items()):
        def sort_key(name):
            stem = PurePosixPath(name).stem
            return (int(stem) if stem.isdigit() else stem)
        selected = sorted(source_names, key=sort_key)[:32]
        target_dir = DATA / (label + "_" + video_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, source in enumerate(selected):
            payload = z.read(source)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None, "Image decode failed: " + source
            suffix = PurePosixPath(source).suffix
            target = target_dir / (str(i).zfill(5) + suffix)
            target.write_bytes(payload)
            relative = "./" + target.relative_to(ROOT).as_posix()
            frames.append(relative)
            audit.append({"source": source, "cloud_path": relative, "label": label, "sha256": hashlib.sha256(payload).hexdigest()})
        manifest["UADFV"][label]["test"][label + "_" + video_id] = {"label": label, "frames": frames}
counts = Counter(label for label, _ in groups)
frame_counts = Counter(a["label"] for a in audit)
assert counts["UADFV_Real"] > 0 and counts["UADFV_Fake"] > 0
report["video_counts"] = dict(counts)
report["frame_counts"] = dict(frame_counts)
report["video_total"] = len(groups)
report["frame_total"] = len(audit)
report["frames_per_video_range"] = [min(len(v["frames"]) for c in manifest["UADFV"].values() for v in c["test"].values()),
                                     max(len(v["frames"]) for c in manifest["UADFV"].values() for v in c["test"].values())]
json_dir = ROOT / "preprocessing/dataset_json_cloud"
json_dir.mkdir(parents=True, exist_ok=True)
(json_dir / "UADFV.json").write_text(json.dumps(manifest, indent=2))
(OUT / "data_audit.json").write_text(json.dumps(audit, indent=2))
print("DATA_READY " + json.dumps({k: report[k] for k in ["video_counts", "frame_counts", "video_total", "frame_total", "archive_bytes"]}), flush=True)

os.chdir(ROOT)
sys.argv = ["test.py"]
test_module = runpy.run_path(str(test_path), run_name="cloud_evaluation_import")
test_config = yaml.safe_load((ROOT / "training/config/test_config.yaml").read_text())
test_config.update({"lmdb": False, "dataset_json_folder": str(json_dir), "rgb_dir": "./datasets/rgb"})
models = [
    ("meso4", "meso4.yaml", "meso4_best.pth"),
    ("meso4Inception", "meso4Inception.yaml", "meso4Incep_best.pth"),
    ("xception", "xception.yaml", "xception_best.pth"),
]
weight_dir = ROOT / "training/weights"
weight_dir.mkdir(exist_ok=True)
for name, cfg_file, weights_file in models:
    print("MODEL_START " + name, flush=True)
    cfg = yaml.safe_load((ROOT / "training/config/detector" / cfg_file).read_text())
    cfg.update(test_config)
    cfg.update({"test_dataset": ["UADFV"], "cuda": False, "pretrained": False, "workers": 2, "test_batchSize": 16})
    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)
    loaders = test_module["prepare_testing_data"](cfg)
    actual = loaders["UADFV"].dataset
    assert len(actual) == len(audit)
    assert len(set(actual.data_dict["image"])) == len(audit)
    weight_url = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/" + weights_file
    response = requests.get(weight_url, timeout=(15, 180))
    response.raise_for_status()
    weights = weight_dir / weights_file
    weights.write_bytes(response.content)
    model = DETECTOR[name](cfg).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    start = time.perf_counter()
    with torch.no_grad():
        result = test_module["test_epoch"](model, loaders)["UADFV"]
    elapsed = time.perf_counter() - start
    predictions = np.asarray(result["pred"])
    labels = np.asarray(result["label"]).astype(int)
    assert predictions.shape == (len(audit),) and np.isfinite(predictions).all()
    record = {k: float(result[k]) for k in ["acc", "auc", "eer", "ap", "video_auc"]}
    record.update({"elapsed_seconds": elapsed, "frames_per_second": len(audit)/elapsed, "checkpoint_sha256": hashlib.sha256(response.content).hexdigest(),
                   "strict_checkpoint_load": True, "parameters": sum(p.numel() for p in model.parameters())})
    report["models"][name] = record
    with (OUT / (name + "_predictions.csv")).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cloud_frame_path", "label_0_real_1_fake", "fake_probability"])
        writer.writerows(zip(actual.data_dict["image"], labels.tolist(), predictions.tolist()))
    (OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    print("MODEL_RESULT " + name + " " + json.dumps(record), flush=True)
    del model, loaders

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
fig, ax = plt.subplots(figsize=(7, 5))
for name in report["models"]:
    with (OUT / (name + "_predictions.csv")).open() as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label_0_real_1_fake"]) for r in rows])
    pred = np.array([float(r["fake_probability"]) for r in rows])
    fpr, tpr, _ = roc_curve(labels, pred)
    ax.plot(fpr, tpr, label=f"{name} (AUC={report['models'][name]['auc']:.4f})")
    report["models"][name]["confusion_matrix_0_1"] = confusion_matrix(labels, (pred > 0.5).astype(int)).tolist()
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"DeepfakeBench / UADFV ({report['frame_total']} face frames)")
ax.legend(loc="lower right"); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig(OUT / "uadfv_roc.png", dpi=180)
(OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2))
print("FINAL_REPORT_START", flush=True)
print(json.dumps(report, indent=2), flush=True)
print("FINAL_REPORT_END", flush=True)
