"""Execute v2 notebooks in fresh kernels; keep sources/output history intact.

python -m scripts.check_notebooks_v2 --run-baselines --run-ablation
CNN training, freeze and evaluation stay disabled. Development kernels receive
a filesystem audit hook that rejects reserved CSVs/images, including indirect reads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import pandas as pd

from scripts.v2_artifacts import repository_root, run_id, write_json
from scripts.v2_data import SPLIT_REL, AUDIT_REL, METADATA_REL


def check_notebooks(run_baselines=False, run_ablation=False):
    root = repository_root()
    output = root / "data/processed/notebook_executions_v2" / run_id("verification")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for source in sorted((root / "notebooks").glob("0*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        if source.name.startswith(("03", "04", "05", "06")):
            reserved = []
            for role in ["calibration", "internal_holdout"]:
                frame = pd.read_csv(root / SPLIT_REL / f"{role}.csv")
                reserved.extend(frame.image_path.tolist())
                reserved.append(f"{SPLIT_REL}/{role}.csv")
            reserved += [f"{SPLIT_REL}/assignments.csv", f"{AUDIT_REL}/image_manifest.csv", f"{AUDIT_REL}/duplicate_pairs.csv",
                         METADATA_REL, "data/raw/lesions/GroundTruth.csv"]
            reserved += [f"data/processed/{role}.csv" for role in ["train", "val", "test"]]
            hook = f'''
from pathlib import Path
import sys
_guard_root = Path.cwd().resolve()
_reserved = set({reserved!r})
_development_reads = set()
def _audit_read(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    try:
        value = args[0].decode() if isinstance(args[0], bytes) else args[0]
        rel = Path(value).resolve().relative_to(_guard_root).as_posix()
    except (ValueError, OSError):
        return
    if rel in _reserved:
        raise AssertionError("Development notebook attempted a reserved read: " + rel)
    if rel.startswith("data/"):
        _development_reads.add(rel)
sys.addaudithook(_audit_read)
'''
            notebook.cells.insert(0, nbformat.v4.new_code_cell(hook))
        if source.name.startswith("03") and not run_baselines:
            for cell in notebook.cells:
                if cell.cell_type == "code": cell.source = cell.source.replace("EXECUTE_BASELINES = True", "EXECUTE_BASELINES = False")
        if source.name.startswith("04") and run_ablation:
            for cell in notebook.cells:
                if cell.cell_type == "code": cell.source = cell.source.replace("EXECUTE_EXPERIMENTS = False", "EXECUTE_EXPERIMENTS = True")
        # No mutation of heavy-work flags, even when all light experiments are requested.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                for flag in ["EXECUTE_TRAINING", "EXECUTE_FREEZE", "EXECUTE_EVALUATION"]:
                    if f"{flag} = True" in cell.source:
                        raise ValueError("Review runner refuses heavy/evaluation execution")
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(root)}})
        start = time.perf_counter()
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / source.name)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        result = {"notebook": source.name, "status": "passed", "seconds": time.perf_counter() - start,
                  "development_filesystem_guard": source.name.startswith(("03", "04", "05", "06")),
                  "source_modified": False}
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(output / "verification.json", {"results": results, "real_cnn_training": False, "real_holdout_evaluation": False})
    print(output.relative_to(root).as_posix(), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    args = parser.parse_args()
    check_notebooks(args.run_baselines, args.run_ablation)
