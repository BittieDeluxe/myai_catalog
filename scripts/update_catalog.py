#!/usr/bin/env python3
"""
Queries HuggingFace for bartowski's top GGUF models under 7B parameters,
then writes catalog.json consumed by the MyAI iOS app.

Run manually:  python3 scripts/update_catalog.py
With HF token: HF_TOKEN=hf_xxx python3 scripts/update_catalog.py
"""

import json
import os
import re
import sys
import requests

API    = "https://huggingface.co/api"
HUB    = "https://huggingface.co"
AUTHOR = "bartowski"
MAX_B  = 7.0   # exclude models >= 7B
TOP_N  = 12    # max entries in catalog
QUANTS = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M"]

HEADERS = {}
if token := os.environ.get("HF_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {token}"

# Curated descriptions and badges for well-known models.
# Keys are the repo name with "-GGUF" and "-Instruct"/"-it" stripped, lowercased.
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


def parse_params_b(text: str) -> float | None:
    """Return parameter count in billions from a model name, or None."""
    # Primary: look for -3B-, _2b_, -1.5B-GGUF style patterns
    m = re.search(r"[-_](\d+\.?\d*)[Bb](?:[-_\s]|GGUF|$)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: any NB pattern in the string
    m = re.search(r"(\d+\.?\d*)[Bb]", text)
    if m:
        val = float(m.group(1))
        if 0.1 <= val <= 200:
            return val
    return None


def best_gguf_file(siblings: list[dict]) -> dict | None:
    """Pick the preferred quantization from the model's file list."""
    for quant in QUANTS:
        for f in siblings:
            name = f.get("rfilename", "")
            if quant.lower() in name.lower() and name.endswith(".gguf"):
                return f
    return None


def nice_name(repo: str) -> str:
    """'Llama-3.2-3B-Instruct-GGUF' -> 'Llama 3.2 3B'"""
    s = re.sub(r"-GGUF$", "", repo, flags=re.IGNORECASE)
    for suffix in ["-Instruct", "-instruct", "-it", "-chat"]:
        s = s.replace(suffix, "")
    # Remove version tags like -v0.3
    s = re.sub(r"-v\d+\.\d+$", "", s)
    s = re.sub(r"[-_]+", " ", s).strip()
    # Capitalize words that start with a lowercase letter (preserve existing casing like SmolLM2)
    s = " ".join(w[0].upper() + w[1:] if w and w[0].islower() else w for w in s.split())
    # Uppercase parameter size suffix (e.g. "2b" -> "2B", "1.5b" -> "1.5B")
    s = re.sub(r"(\d+\.?\d*)([Bb])\b", lambda m: m.group(1) + "B", s)
    return s


def make_id(repo: str) -> str:
    """Stable lowercase ID, e.g. 'llama-3.2-3b-instruct-gguf' -> 'llama-3.2-3b-q4'"""
    s = repo.lower()
    s = re.sub(r"-gguf$", "", s)
    s = re.sub(r"-instruct|-it|-chat", "", s)
    s = re.sub(r"-v\d+\.\d+$", "", s)
    return s + "-q4"


def meta_key(repo: str) -> str:
    """Match METADATA keys: strip -GGUF, -Instruct, lowercase."""
    s = re.sub(r"-GGUF$", "", repo, flags=re.IGNORECASE)
    for suffix in ["-Instruct", "-instruct", "-it", "-chat"]:
        s = s.replace(suffix, "")
    return s.lower()


def fetch_catalog() -> list[dict]:
    print(f"Listing {AUTHOR}'s GGUF models by downloads...")
    resp = requests.get(
        f"{API}/models",
        params={
            "author": AUTHOR,
            "filter": "gguf",
            "sort": "downloads",
            "direction": -1,
            "limit": 100,
        },
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    all_models = resp.json()
    print(f"  Got {len(all_models)} repos")

    catalog = []
    seen_base = set()

    for entry in all_models:
        if len(catalog) >= TOP_N:
            break

        model_id = entry.get("id", "")       # "bartowski/Llama-3.2-3B-Instruct-GGUF"
        repo     = model_id.split("/")[-1]   # "Llama-3.2-3B-Instruct-GGUF"

        param_b = parse_params_b(repo)
        if param_b is None:
            print(f"  skip (unparseable size): {repo}")
            continue
        if param_b >= MAX_B:
            print(f"  skip ({param_b}B >= {MAX_B}B): {repo}")
            continue

        # Deduplicate: same base model shouldn't appear twice
        base = re.sub(r"[-_](?:instruct|it|chat|v\d.*|gguf).*", "", repo.lower())
        if base in seen_base:
            print(f"  skip (duplicate base): {repo}")
            continue

        # Fetch file list
        detail = requests.get(f"{API}/models/{model_id}", headers=HEADERS, timeout=30)
        if not detail.ok:
            print(f"  skip (detail fetch failed {detail.status_code}): {repo}")
            continue

        siblings  = detail.json().get("siblings", [])
        gguf_file = best_gguf_file(siblings)
        if not gguf_file:
            print(f"  skip (no Q4 GGUF found): {repo}")
            continue

        filename   = gguf_file["rfilename"]
        size_bytes = gguf_file.get("size", 0)
        size_gb    = round(size_bytes / 1_073_741_824, 2) if size_bytes else round(param_b * 0.55, 2)
        ram_gb     = round(size_gb * 1.3 + 0.3, 1)
        mk         = meta_key(repo)
        meta       = METADATA.get(mk, {})

        catalog.append({
            "id":            make_id(repo),
            "name":          nice_name(repo),
            "description":   meta.get("description", f"{param_b}B parameter model, Q4 quantized."),
            "sizeGB":        size_gb,
            "ramRequiredGB": ram_gb,
            "downloadURL":   f"{HUB}/{model_id}/resolve/main/{filename}",
            "filename":      filename,
            "badge":         meta.get("badge"),
        })
        seen_base.add(base)
        print(f"  + {nice_name(repo)} ({param_b}B, {size_gb} GB)")

    return catalog


def main() -> None:
    print(f"Building catalog: top {TOP_N} models under {MAX_B}B from {AUTHOR}\n")

    try:
        catalog = fetch_catalog()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Keeping existing catalog.json unchanged.")
        sys.exit(0)  # Don't fail the workflow — stale catalog is better than no catalog

    if len(catalog) < 3:
        print(f"\nSanity check failed: only {len(catalog)} models found. Aborting.")
        sys.exit(0)

    with open("catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nWrote {len(catalog)} models to catalog.json")


if __name__ == "__main__":
    main()
