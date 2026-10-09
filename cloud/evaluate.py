"""Evaluate three released DeepfakeBench detectors on a fixed official DeepFakeFace image subset.
All project execution, weights, and dataset processing happen on a GitHub cloud runner.
"""
import os, sys, json, time, hashlib, platform, csv, random, warnings, runpy
from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import requests, cv2, numpy as np, torch, yaml
from PIL import Image

WORK = Path.cwd().resolve()
ROOT = WORK / "DeepfakeBench"
OUT = WORK / "results"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "datasets/rgb/DeepFakeFace_cloud"
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
    "python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    "upstream_commit": "f188b1c105465e2e5377eb536a95022ae0e4522d",
    "dataset": "DeepFakeFace paired image subset (wiki real / insight fake)",
    "dataset_source": "Official OpenRL/DeepFakeFace release",
    "data_url": "https://huggingface.co/datasets/OpenRL/DeepFakeFace",
    "scope": "Pretrained detector inference and metrics on a fixed paired subset; no training; this is not the official UADFV or FF++ benchmark reproduction",
    "blocked_original_dataset": "Author UADFV Google Drive archive exceeded download quota; logged in workflow run 37919114217",
    "preprocessing": "Original released images, RGB cubic resize to 256x256 and normalization from original DeepfakeAbstractBaseDataset; no extra face cropping",
    "resolution": [256, 256], "normalization_mean": [0.5]*3, "normalization_std": [0.5]*3,
    "seed": 1024, "batch_size": 16, "cpu_threads": 4,
    "adaptations": [
        "Select only evaluated detectors, networks, losses, and dataset imports",
        "Remove unused training-only imports from test.py",
        "Skip redundant ImageNet backbone initialization before strict complete Xception checkpoint load",
        "Generate a dataset JSON for the paired image subset; use a unique sample folder per image"
    ],
    "evaluation_functions": "Original training/test.py prepare_testing_data and test_epoch; original DeepfakeAbstractBaseDataset and metrics/utils.py get_test_metrics",
    "metric_notes": "Each sample contains one image, so video_auc is identical to frame AUC and must not be interpreted as a video benchmark",
    "models": {}
}
(OUT / "run_manifest.json").write_text(json.dumps(report, indent=2))
from concurrent.futures import ThreadPoolExecutor
import zipfile
# Pin the public dataset snapshot for reproducible file selection.
info = requests.get("https://huggingface.co/api/datasets/OpenRL/DeepFakeFace", timeout=(15, 45))
info.raise_for_status()
dataset_sha = info.json()["sha"]
report["dataset_revision"] = dataset_sha

def download_archive(name):
    url = f"https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/{dataset_sha}/{name}?download=true"
    target = ROOT / "datasets" / name
    print("DATA_DOWNLOAD_START " + name, flush=True)
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        size = 0
        previous = time.monotonic()
        with target.open("wb") as f:
            for block in response.iter_content(chunk_size=1024*1024):
                if not block:
                    continue
                f.write(block); digest.update(block); size += len(block)
                if time.monotonic() - previous > 20:
                    print(f"DATA_DOWNLOAD_PROGRESS {name} {size/1024/1024:.1f} MiB", flush=True)
                    previous = time.monotonic()
    assert zipfile.is_zipfile(target), "Downloaded file is not a ZIP: " + name
    print(f"DATA_DOWNLOAD_DONE {name} {size} bytes", flush=True)
    return name, target, size, digest.hexdigest()
with ThreadPoolExecutor(max_workers=2) as pool:
    archives = list(pool.map(download_archive, ["wiki.zip", "insight.zip"]))
report["archives"] = {name: {"bytes": size, "sha256": sha} for name, _, size, sha in archives}
paths = {name: path for name, path, _, _ in archives}
def image_names(archive):
    result = {}
    for name in archive.namelist():
        if name.startswith("__MACOSX/") or not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        key = PurePosixPath(name).stem
        assert key not in result, "Duplicate sample identifier: " + key
        result[key] = name
    return result

with zipfile.ZipFile(paths["wiki.zip"]) as real_zip, zipfile.ZipFile(paths["insight.zip"]) as fake_zip:
    real_map, fake_map = image_names(real_zip), image_names(fake_zip)
    common = sorted(set(real_map) & set(fake_map))
    assert len(common) >= 256, f"Too few paired samples: {len(common)}"
    selected = sorted(random.Random(1024).sample(common, 256))
    report["available_real_images"] = len(real_map)
    report["available_fake_images"] = len(fake_map)
    report["available_paired_ids"] = len(common)
    report["selected_pair_ids"] = selected
    manifest = {"DeepFakeFace_subset": {label: {"test": {}} for label in ["DFF_Real", "DFF_Fake"]}}
    audit = []
    for label, archive, mapping in [("DFF_Real", real_zip, real_map), ("DFF_Fake", fake_zip, fake_map)]:
        for sample_id in selected:
            source = mapping[sample_id]
            payload = archive.read(source)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None, "Image decode failed: " + source
            target_dir = DATA / (label + "_" + sample_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / ("00000" + PurePosixPath(source).suffix)
            target.write_bytes(payload)
            relative = "./" + target.relative_to(ROOT).as_posix()
            manifest["DeepFakeFace_subset"][label]["test"][label + "_" + sample_id] = {"label": label, "frames": [relative]}
            audit.append({"archive": "wiki.zip" if label == "DFF_Real" else "insight.zip", "source": source,
                          "cloud_path": relative, "label": label, "sha256": hashlib.sha256(payload).hexdigest(), "source_shape": list(image.shape)})
report["image_counts"] = {"real": 256, "fake": 256}
report["pair_total"] = len(selected)
report["frame_total"] = len(audit)
report["frame_counts"] = dict(Counter(a["label"] for a in audit))
json_dir = ROOT / "preprocessing/dataset_json_cloud"
json_dir.mkdir(parents=True, exist_ok=True)
(json_dir / "DeepFakeFace_subset.json").write_text(json.dumps(manifest, indent=2))
(OUT / "data_audit.json").write_text(json.dumps(audit, indent=2))
print("DATA_READY " + json.dumps({k: report[k] for k in ["image_counts", "pair_total", "frame_total", "archives"]}), flush=True)

os.chdir(ROOT)
sys.argv = ["test.py"]
test_module = runpy.run_path(str(test_path), run_name="cloud_evaluation_import")
test_config = yaml.safe_load((ROOT / "training/config/test_config.yaml").read_text())
test_config["label_dict"].update({"DFF_Real": 0, "DFF_Fake": 1})
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
    cfg.update({"test_dataset": ["DeepFakeFace_subset"], "cuda": False, "pretrained": False, "workers": 2, "test_batchSize": 16})
    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)
    loaders = test_module["prepare_testing_data"](cfg)
    actual = loaders["DeepFakeFace_subset"].dataset
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
        result = test_module["test_epoch"](model, loaders)["DeepFakeFace_subset"]
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
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"DeepfakeBench / DeepFakeFace subset ({report['frame_total']} images)")
ax.legend(loc="lower right"); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig(OUT / "deepfakeface_roc.png", dpi=180)
(OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2))
print("FINAL_REPORT_START", flush=True)
print(json.dumps(report, indent=2), flush=True)
print("FINAL_REPORT_END", flush=True)
