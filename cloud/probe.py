import os, sys, json, time, hashlib, platform, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import torch
import yaml

ROOT = Path("DeepfakeBench").resolve()
OUT = Path("results")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(1024)
report = {
    "execution_location": "GitHub-hosted ubuntu-22.04 cloud runner",
    "platform": platform.platform(),
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "upstream_commit": "f188b1c105465e2e5377eb536a95022ae0e4522d",
    "evaluation_scope": "Pretrained checkpoint smoke test; no accuracy claims on synthetic input",
}

# Select only the requested detector imports. Architecture and forward code are unchanged.
(ROOT / "training/detectors/__init__.py").write_text("from metrics.registry import DETECTOR\nfrom .meso4_detector import Meso4Detector\n")
(ROOT / "training/networks/__init__.py").write_text("from metrics.registry import BACKBONE\nfrom .mesonet import Meso4, MesoInception4\n")
(ROOT / "training/loss/__init__.py").write_text("from metrics.registry import LOSSFUNC\nfrom .cross_entropy_loss import CrossEntropyLoss\n")
sys.path.insert(0, str(ROOT / "training"))
from detectors import DETECTOR
config = yaml.safe_load((ROOT / "training/config/detector/meso4.yaml").read_text())
config["cuda"] = False
model = DETECTOR["meso4"](config).eval()
weights = OUT / "meso4_best.pth"
url = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/meso4_best.pth"
response = requests.get(url, timeout=(15, 90))
response.raise_for_status()
weights.write_bytes(response.content)
report["checkpoint_sha256"] = hashlib.sha256(response.content).hexdigest()
model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
report["strict_checkpoint_load"] = True
report["parameters"] = sum(p.numel() for p in model.parameters())
batch = torch.rand(4, 3, 256, 256) * 2 - 1
start = time.perf_counter()
with torch.no_grad():
    predictions = model({"image": batch}, inference=True)
elapsed = time.perf_counter() - start
assert predictions["cls"].shape == (4, 2)
assert predictions["prob"].shape == (4,)
assert torch.isfinite(predictions["prob"]).all()
report["smoke"] = {
    "input_shape": list(batch.shape),
    "logits_shape": list(predictions["cls"].shape),
    "probabilities": predictions["prob"].tolist(),
    "elapsed_seconds": elapsed,
    "passed": True
}

folder_id = "1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs"
try:
    response = requests.get("https://drive.google.com/embeddedfolderview", params={"id": folder_id}, timeout=(15, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [{"name": a.get_text(strip=True), "url": a.get("href")} for a in soup.find_all("a") if a.get("href")]
    report["data_folder_links"] = links
    (OUT / "data_folder_links.json").write_text(json.dumps(links, ensure_ascii=False, indent=2))
except Exception as exc:
    report["data_folder_error"] = f"{type(exc).__name__}: {exc}"
(OUT / "probe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
