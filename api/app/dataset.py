from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

from app.settings import Settings


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

API_DIR = Path(__file__).resolve().parents[1]  # .../pimple/api

# Cache simples do CSV (evita reler toda hora)
_CSV_CACHE_KEY: Optional[Tuple[str, float]] = None
_CSV_CACHE_INDEX: Dict[str, dict] = {}
_CSV_CACHE_COLUMNS: List[str] = []


# Nomes bonitinhos (padrão HAM10000)
CLASS_PRETTY = {
    "MEL": "Melanoma",
    "NV": "Nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
}


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
    """
    Gera possíveis IDs equivalentes para casar imagem <-> máscara.
    Ex:
      ISIC_0024306_segmentation -> ISIC_0024306
    """
    s = stem.strip()
    aliases = {s}

    # sufixos comuns
    for suf in ["_mask", "-mask", " mask", "_seg", "-seg", "_segmentation", "-segmentation"]:
        if s.lower().endswith(suf):
            aliases.add(s[: -len(suf)])

    # prefixos comuns
    if s.lower().startswith("mask_"):
        aliases.add(s[5:])
    if s.lower().startswith("mask-"):
        aliases.add(s[5:])

    out = []
    for a in aliases:
        a2 = a.strip()
        if a2 and a2 not in out:
            out.append(a2)
    return out


def _normalize_id(value: str) -> str:
    """
    Normaliza ID vindo do CSV:
    - remove extensão se vier "ISIC_XXXX.jpg"
    - remove path se vier "images/ISIC_XXXX"
    """
    s = (value or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    last = s.split("/")[-1]
    # remove extensão se existir
    return Path(last).stem


def _to_number_if_possible(x: str):
    s = (x or "").strip()
    if s == "":
        return s
    # tenta int
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    # tenta float
    try:
        return float(s)
    except Exception:
        return s


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


def _load_csv_index(settings: Settings) -> Tuple[Dict[str, dict], List[str]]:
    """
    Carrega o CSV e cria um índice: id -> meta (dict com colunas)
    Cacheado por (path_resolvido, mtime)
    """
    global _CSV_CACHE_KEY, _CSV_CACHE_INDEX, _CSV_CACHE_COLUMNS

    csv_path = _resolve_path(settings.dataset_csv_path)
    if not settings.dataset_csv_path or not csv_path.exists() or not csv_path.is_file():
        _CSV_CACHE_KEY = None
        _CSV_CACHE_INDEX = {}
        _CSV_CACHE_COLUMNS = []
        return _CSV_CACHE_INDEX, _CSV_CACHE_COLUMNS

    key = (str(csv_path), csv_path.stat().st_mtime)
    if _CSV_CACHE_KEY == key:
        return _CSV_CACHE_INDEX, _CSV_CACHE_COLUMNS

    # (re)carrega
    index: Dict[str, dict] = {}
    columns: List[str] = []

    # tenta utf-8 e fallback latin-1
    encodings = ["utf-8", "latin-1"]
    last_error = None
    for enc in encodings:
        try:
            with csv_path.open("r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                # tenta achar coluna "image" (ou parecidas)
                image_col = None
                for c in columns:
                    if c and c.strip().lower() in ("image", "image_id", "img", "filename", "file", "id"):
                        image_col = c
                        break
                # se não achou, usa primeira coluna
                if not image_col and columns:
                    image_col = columns[0]

                for row in reader:
                    raw_id = row.get(image_col, "") if image_col else ""
                    item_id = _normalize_id(raw_id)
                    if not item_id:
                        continue

                    meta = {}
                    for k, v in row.items():
                        if k is None:
                            continue
                        if image_col and k == image_col:
                            continue
                        meta[k] = _to_number_if_possible(v)

                    index[item_id] = meta

            last_error = None
            break
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        # se falhou pra valer, zera (mas não quebra o server)
        index = {}
        columns = []

    _CSV_CACHE_KEY = key
    _CSV_CACHE_INDEX = index
    _CSV_CACHE_COLUMNS = columns
    return _CSV_CACHE_INDEX, _CSV_CACHE_COLUMNS


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


def _enrich_meta_with_target(meta: dict) -> dict:
    """
    Se o CSV for one-hot (MEL/NV/BCC...), cria:
      - positive_classes
      - target
      - target_pretty
    """
    if not meta:
        return {}

    positives: List[str] = []
    for k, v in meta.items():
        try:
            if isinstance(v, (int, float)) and float(v) == 1.0:
                positives.append(str(k))
        except Exception:
            continue

    target = positives[0] if len(positives) == 1 else ("|".join(positives) if positives else "")
    target_pretty = CLASS_PRETTY.get(target, target)

    enriched = dict(meta)
    enriched["positive_classes"] = positives
    enriched["target"] = target
    enriched["target_pretty"] = target_pretty
    return enriched


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

    csv_index, _ = _load_csv_index(settings)
    csv_meta = csv_index.get(item_id, {})
    meta = _enrich_meta_with_target(csv_meta)

    return {
        "id": item_id,
        "image_filename": img.name if img else None,
        "mask_filename": mask.name if mask else None,
        "has_mask": mask is not None,
        "meta": meta,
    }


def get_image_file(settings: Settings, item_id: str) -> Optional[str]:
    images_by_id, _ = _index_files(settings)
    fp = images_by_id.get(item_id)
    return str(fp) if fp else None


def get_mask_file(settings: Settings, item_id: str) -> Optional[str]:
    _, masks_by_id = _index_files(settings)
    fp = masks_by_id.get(item_id)
    return str(fp) if fp else None
