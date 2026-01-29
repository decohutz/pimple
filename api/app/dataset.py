from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

from app.settings import Settings


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".webp"}  # depende do dataset; deixamos amplo


def _safe_list(dir_path: str, exts: set[str]) -> List[Path]:
    if not dir_path:
        return []
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return []
    files = []
    for fp in p.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts:
            files.append(fp)
    files.sort(key=lambda x: x.name.lower())
    return files


def _stem_id(path: Path) -> str:
    return path.stem


def _read_csv_header_and_rows(csv_path: str) -> Tuple[List[str], int]:
    if not csv_path:
        return [], 0
    p = Path(csv_path)
    if not p.exists() or not p.is_file():
        return [], 0

    try:
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
            return header, rows
    except UnicodeDecodeError:
        # fallback comum em CSVs
        with p.open("r", newline="", encoding="latin-1") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
            return header, rows
    except Exception:
        return [], 0


def _index_files(settings: Settings) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images = _safe_list(settings.dataset_images_dir, IMAGE_EXTS)
    masks = _safe_list(settings.dataset_masks_dir, MASK_EXTS)

    images_by_id: Dict[str, Path] = {}
    masks_by_id: Dict[str, Path] = {}

    for fp in images:
        images_by_id[_stem_id(fp)] = fp

    for fp in masks:
        masks_by_id[_stem_id(fp)] = fp

    return images_by_id, masks_by_id


def get_dataset_summary(settings: Settings) -> dict:
    images_by_id, masks_by_id = _index_files(settings)
    header, rows = _read_csv_header_and_rows(settings.dataset_csv_path)

    return {
        "num_images": len(images_by_id),
        "num_masks": len(masks_by_id),
        "csv_rows": int(rows),
        "columns": header,
    }


def list_dataset_items(settings: Settings, limit: int, offset: int, query: str) -> dict:
    images_by_id, masks_by_id = _index_files(settings)
    ids = sorted(images_by_id.keys())

    q = (query or "").strip().lower()
    if q:
        ids = [i for i in ids if q in i.lower()]

    total = len(ids)
    page = ids[offset : offset + limit]

    items = [{"id": i, "has_mask": i in masks_by_id} for i in page]
    return {"items": items, "total": total}


def get_dataset_item(settings: Settings, item_id: str) -> Optional[dict]:
    images_by_id, masks_by_id = _index_files(settings)
    img = images_by_id.get(item_id)
    if not img:
        return None

    mask = masks_by_id.get(item_id)
    return {
        "id": item_id,
        "image_filename": img.name if img else None,
        "mask_filename": mask.name if mask else None,
        "has_mask": mask is not None,
        "meta": {},  # reservado para no futuro puxar colunas do CSV
    }


def get_image_file(settings: Settings, item_id: str) -> Optional[str]:
    images_by_id, _ = _index_files(settings)
    fp = images_by_id.get(item_id)
    return str(fp) if fp else None


def get_mask_file(settings: Settings, item_id: str) -> Optional[str]:
    _, masks_by_id = _index_files(settings)
    fp = masks_by_id.get(item_id)
    return str(fp) if fp else None
