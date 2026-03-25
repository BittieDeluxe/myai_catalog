#!/usr/bin/env python3
"""
Queries HuggingFace for bartowski's top GGUF models under 7B parameters,
then writes catalog.json consumed by the MyAI iOS app.
Includes multiple quantization variants per model.

Run manually:  python3 scripts/update_catalog.py
With HF token: HF_TOKEN=hf_xxx python3 scripts/update_catalog.py
"""

import json
import os
import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API    = "https://huggingface.co/api"
HUB    = "https://huggingface.co"
AUTHOR = "bartowski"
MAX_B  = 7.0   # exclude models >= 7B params
TOP_N  = 12    # max base models in catalog (all quants for each are included)
QUANTS = ["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q4_0", "Q5_K_M", "Q6_K", "Q8_0"]

# Estimated GB per billion parameters — used when the API returns no file size.
GB_PER_PARAM: dict[str, float] = {
    "Q2_K":   0.31,
    "Q3_K_M": 0.42,
    "Q4_K_S": 0.52,
    "Q4_0":   0.51,
    "Q4_K_M": 0.55,
    "Q5_K_M": 0.65,
    "Q6_K":   0.75,
    "Q8_0":   0.87,
}

# ID suffix per quant. Q4_K_M keeps "q4" for backward compatibility with
# existing user saved-model IDs already persisted on devices.
QUANT_SUFFIX: dict[str, str] = {
    "Q2_K":   "q2",
    "Q3_K_M": "q3",
    "Q4_K_S": "q4ks",
    "Q4_K_M": "q4",
    "Q4_0":   "q40",
    "Q5_K_M": "q5",
    "Q6_K":   "q6",
    "Q8_0":   "q8",
}

HEADERS: dict[str, str] = {}
if token := os.environ.get("HF_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {token}"

CATEGORIES = [
    {"name": "Reasoning",  "icon": "brain.head.profile",                     "search": "R1"},
    {"name": "Coding",     "icon": "chevron.left.forwardslash.chevron.right", "search": "Coder"},
    {"name": "General",    "icon": "bubble.left.and.bubble.right",            "search": "Llama"},
    {"name": "Creative",   "icon": "pencil.and.sparkles",                     "search": "Hermes"},
]

# Curated descriptions and badges for well-known models (keyed by normalised repo name).
METADATA: dict[str, dict] = {
    "gemma-2-2b-it": {
        "description": "Google's compact model. Fastest option, great for quick tasks.",
        "badge": "Fastest",
    },
    "llama-3.2-3b-instruct": {
        "description": "Meta's efficient 3B model. Great balance of speed and quality.",
        "badge": None,
    },
    "phi-3.5-mini-instruct": {
        "description": "Microsoft's Phi-3.5 Mini. Excellent reasoning for its size.",
        "badge": "Recommended",
    },
    "qwen_qwen3.5-2b": {
        "description": "Alibaba's Qwen3.5 2B. Excellent quality, supports 200+ languages.",
        "badge": None,
    },
    "qwen_qwen3.5-4b": {
        "description": "Alibaba's Qwen3.5 4B. Beats models twice its size on reasoning.",
        "badge": None,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_params_b(text: str) -> float | None:
    """Return parameter count in billions from a model name, or None."""
    m = re.search(r"[-_](\d+\.?\d*)[Bb](?:[-_\s]|GGUF|$)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)[Bb]", text)
    if m:
        val = float(m.group(1))
        if 0.1 <= val <= 200:
            return val
    return None


def all_gguf_files(siblings: list[dict]) -> dict[str, dict]:
    """Return {quant: file_dict} for every quant present in the repo."""
    result: dict[str, dict] = {}
    for quant in QUANTS:
        for f in siblings:
            name = f.get("rfilename", "")
            if quant.lower() in name.lower() and name.endswith(".gguf"):
                result[quant] = f
                break
    return result


def badge_for_quant(quant: str, model_badge: str | None) -> str | None:
    """Assign a display badge based on quantization level."""
    if quant == "Q8_0":
        return "High Quality"
    if quant == "Q3_K_M":
        return "Most Compatible"
    if quant in ("Q4_K_M", "Q4_K_S", "Q4_0"):
        return model_badge  # use the model's curated badge if any
    return None


def nice_name(repo: str) -> str:
    s = re.sub(r"-GGUF$", "", repo, flags=re.IGNORECASE)
    for suffix in ["-Instruct", "-instruct", "-it", "-chat"]:
        s = s.replace(suffix, "")
    s = re.sub(r"-v\d+\.\d+$", "", s)
    s = re.sub(r"[-_]+", " ", s).strip()
    s = " ".join(w[0].upper() + w[1:] if w and w[0].islower() else w for w in s.split())
    s = re.sub(r"(\d+\.?\d*)([Bb])\b", lambda m: m.group(1) + "B", s)
    return s


def make_id(repo: str, quant: str) -> str:
    s = repo.lower()
    s = re.sub(r"-gguf$", "", s)
    s = re.sub(r"-instruct|-it|-chat", "", s)
    s = re.sub(r"-v\d+\.\d+$", "", s)
    return s + "-" + QUANT_SUFFIX.get(quant, quant.lower())


def meta_key(repo: str) -> str:
    s = re.sub(r"-GGUF$", "", repo, flags=re.IGNORECASE)
    for suffix in ["-Instruct", "-instruct", "-it", "-chat"]:
        s = s.replace(suffix, "")
    return s.lower()


def build_variant(
    model_id: str,
    repo: str,
    quant: str,
    gguf_file: dict,
    param_b: float | None,
    hf_downloads: int = 0,
) -> dict | None:
    """Build one ModelDefinition-compatible dict for a specific quant variant."""
    fname      = gguf_file["rfilename"]
    size_bytes = gguf_file.get("size")

    if size_bytes:
        size_gb = round(size_bytes / 1_073_741_824, 2)
    elif param_b:
        size_gb = round(param_b * GB_PER_PARAM.get(quant, 0.55), 1)
    else:
        return None

    ram_gb = round(size_gb * 1.3 + 0.3, 1)
    mk     = meta_key(repo)
    meta   = METADATA.get(mk, {})

    # Use curated description for the standard Q4_K_M variant; generic for others.
    if quant == "Q4_K_M" and meta.get("description"):
        description = meta["description"]
    elif param_b:
        p_str = str(int(param_b)) + "B" if param_b == int(param_b) else str(param_b) + "B"
        description = f"{p_str} parameter model, quantized to {quant}."
    else:
        description = f"Quantized to {quant}."

    return {
        "id":            make_id(repo, quant),
        "name":          nice_name(repo),
        "description":   description,
        "sizeGB":        size_gb,
        "ramRequiredGB": ram_gb,
        "downloadURL":   f"{HUB}/{model_id}/resolve/main/{fname}",
        "filename":      fname,
        "badge":         badge_for_quant(quant, meta.get("badge")),
        "hfDownloads":   hf_downloads,
        "quantization":  quant,
    }


def build_model_variants(model_id: str) -> list[dict]:
    """Fetch model detail and return all available quant variants."""
    try:
        r = requests.get(f"{API}/models/{model_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        detail = r.json()
    except Exception:
        return []

    repo         = model_id.split("/")[-1]
    param_b      = parse_params_b(repo)
    hf_downloads = detail.get("downloads", 0)
    quant_files  = all_gguf_files(detail.get("siblings", []))

    variants = []
    for quant, gguf_file in quant_files.items():
        v = build_variant(model_id, repo, quant, gguf_file, param_b, hf_downloads)
        if v:
            variants.append(v)
    return variants


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def fetch_catalog() -> list[dict]:
    print(f"Listing {AUTHOR}'s GGUF models by downloads...")
    resp = requests.get(
        f"{API}/models",
        params={"author": AUTHOR, "filter": "gguf",
                "sort": "downloads", "direction": -1, "limit": 100},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    all_models = resp.json()
    print(f"  Got {len(all_models)} repos")

    catalog: list[dict] = []
    seen_base: set[str] = set()
    base_count = 0

    for entry in all_models:
        if base_count >= TOP_N:
            break

        model_id = entry.get("id", "")
        repo     = model_id.split("/")[-1]
        param_b  = parse_params_b(repo)

        if param_b is None:
            print(f"  skip (unparseable size): {repo}")
            continue
        if param_b >= MAX_B:
            print(f"  skip ({param_b}B >= {MAX_B}B): {repo}")
            continue

        base = re.sub(r"[-_](?:instruct|it|chat|v\d.*|gguf).*", "", repo.lower())
        if base in seen_base:
            print(f"  skip (duplicate base): {repo}")
            continue

        detail = requests.get(f"{API}/models/{model_id}", headers=HEADERS, timeout=30)
        if not detail.ok:
            print(f"  skip (detail fetch failed {detail.status_code}): {repo}")
            continue

        quant_files = all_gguf_files(detail.json().get("siblings", []))
        if not quant_files:
            print(f"  skip (no matching GGUF found): {repo}")
            continue

        hf_downloads = entry.get("downloads", 0)
        added = 0
        for quant, gguf_file in quant_files.items():
            v = build_variant(model_id, repo, quant, gguf_file, param_b, hf_downloads)
            if v:
                catalog.append(v)
                added += 1

        if added:
            print(f"  + {nice_name(repo)} — {added} variant(s)")
            seen_base.add(base)
            base_count += 1

    return catalog


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def fetch_category(name: str, icon: str, search: str) -> dict:
    """Fetch top 5 base models for a category, all quant variants for each."""
    params = {"author": AUTHOR, "filter": "gguf",
              "sort": "downloads", "direction": "-1", "limit": "8"}
    if search:
        params["search"] = search
    try:
        r = requests.get(f"{API}/models", headers=HEADERS, timeout=10, params=params)
        r.raise_for_status()
        summaries = r.json()
    except Exception as e:
        print(f"Warning: could not fetch {name} category: {e}", file=sys.stderr)
        return {"name": name, "icon": icon, "models": []}

    all_variants: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(build_model_variants, s["id"]): s["id"] for s in summaries}
        for future in as_completed(futures):
            all_variants.extend(future.result())

    # Group by model name, keep top 5 base models by download count
    by_name: dict[str, list[dict]] = {}
    for v in all_variants:
        by_name.setdefault(v["name"], []).append(v)

    name_downloads = {
        n: max(v.get("hfDownloads", 0) for v in vs)
        for n, vs in by_name.items()
    }
    top_names = sorted(name_downloads, key=name_downloads.get, reverse=True)[:5]  # type: ignore[arg-type]

    models: list[dict] = []
    for n in top_names:
        models.extend(by_name[n])

    return {"name": name, "icon": icon, "models": models}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building catalog: top {TOP_N} base models under {MAX_B}B from {AUTHOR}\n")

    try:
        catalog = fetch_catalog()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Keeping existing catalog.json unchanged.")
        sys.exit(0)  # Don't fail the workflow — stale catalog is better than no catalog

    if len(catalog) < 3:
        print(f"\nSanity check failed: only {len(catalog)} models found. Aborting.")
        sys.exit(0)

    print("\nFetching category models...")
    categories: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(fetch_category, c["name"], c["icon"], c["search"]): c["name"]
            for c in CATEGORIES
        }
        for future in as_completed(futures):
            categories.append(future.result())

    order = {c["name"]: i for i, c in enumerate(CATEGORIES)}
    categories.sort(key=lambda c: order.get(c["name"], 99))

    output = {"models": catalog, "categories": categories}
    with open("catalog.json", "w") as f:
        json.dump(output, f, indent=2)

    variant_count = len(catalog)
    base_count    = len({m["name"] for m in catalog})
    cat_count     = sum(len(c["models"]) for c in categories)
    print(f"\nWrote {variant_count} variants across {base_count} base models + {cat_count} category variants to catalog.json")


if __name__ == "__main__":
    main()
