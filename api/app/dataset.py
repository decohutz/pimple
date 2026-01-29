from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

from app.settings import Settings


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

API_DIR = Path(__file__).resolve().parents[1]  # .../pimple/api


def _resolve_path(p: str) -> Path:
    """
    Se p for relativo, resolve a partir de pimple/api/.
    Se for absoluto, mantém.
    """
    if not p:
        return Path("")
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (API_DIR / pp).resolve()


def _safe_list(dir_path: str, exts: set[str]) -> List[Path]:
    p = _resolve_path(dir_path)
    if not dir_path or not p.exists() or not p.is_dir():
        return []
    files = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in exts]
    files.sort(key=lambda x: x.name.lower())
    return files


def _aliases_from_stem(stem: str) -> List[str]:
    s = stem
    aliases = {s}

    for suf in ["_mask", "-mask", " mask", "_seg", "-seg", "_segmentation", "-segmentation"]:
        if s.lower().endswith(suf):
            aliases.add(s[: -len(suf)])

    if s.lower().startswith("mask_"):
        aliases.add(s[5:])
    if s.lower().startswith("mask-"):
        aliases.add(s[5:])

    aliases.add(s.strip())

    out = []
    for a in aliases:
        a2 = a.strip()
        if a2 and a2 not in out:
            out.append(a2)
    return out


def _read_csv_header_and_rows(csv_path: str) -> Tuple[List[str], int]:
    p = _resolve_path(csv_path)
    if not csv_path or not p.exists() or not p.is_file():
        return [], 0

    def _read(enc: str) -> Tuple[List[str], int]:
        with p.open("r", newline="", encoding=enc) as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
            return header, rows

    try:
        return _read("utf-8")
    except UnicodeDecodeError:
        try:
            return _read("latin-1")
        except Exception:
            return [], 0
    except Exception:
        return [], 0


def _index_files(settings: Settings) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images = _safe_list(settings.dataset_images_dir, IMAGE_EXTS)
    masks = _safe_list(settings.dataset_masks_dir, MASK_EXTS)

    images_by_id: Dict[str, Path] = {}
    masks_by_id: Dict[str, Path] = {}

    for fp in images:
        images_by_id[fp.stem] = fp

    for fp in masks:
        for alias in _aliases_from_stem(fp.stem):
            masks_by_id.setdefault(alias, fp)

    return images_by_id, masks_by_id


def get_dataset_summary(settings: Settings) -> dict:
    images_by_id, masks_by_id = _index_files(settings)
    header, rows = _read_csv_header_and_rows(settings.dataset_csv_path)
    return {
        "num_images": len(images_by_id),
        "num_masks": len(set(masks_by_id.values())),
        "csv_rows": int(rows),
        "columns": header,
    }


def list_dataset_items(settings: Settings, limit: int, offset: int, query: str) -> dict:
    images_by_id, masks_by_id = _index_files(settings)

    ids = sorted(images_by_id.keys(), key=lambda x: x.lower())
    q = (query or "").strip().lower()
    if q:
        ids = [i for i in ids if q in i.lower()]

    total = len(ids)
    page = ids[offset: offset + limit]

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
        "meta": {},
    }


def get_image_file(settings: Settings, item_id: str) -> Optional[str]:
    images_by_id, _ = _index_files(settings)
    fp = images_by_id.get(item_id)
    return str(fp) if fp else None


def get_mask_file(settings: Settings, item_id: str) -> Optional[str]:
    _, masks_by_id = _index_files(settings)
    fp = masks_by_id.get(item_id)
    return str(fp) if fp else None
