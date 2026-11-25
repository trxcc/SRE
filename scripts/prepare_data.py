#!/usr/bin/env python3
"""Materialize public Hugging Face datasets (trxcc2002/DEAL-RS & trxcc2002/DEAL-WS)
into the on-disk task layout consumed by DEAL training and evaluation.

Examples:
  # Prepare DEAL-RS (1,800 train / 200 test)
  python3 scripts/prepare_data.py --benchmark deal-rs --output ./data/deal-rs

  # Prepare DEAL-WS (2,000 train / 200 test) using Workspace-Bench filesys_en.zip
  python3 scripts/prepare_data.py --benchmark deal-ws --output ./data/deal-ws \
      --workspace-zip /path/to/filesys_en.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath


HF_REPOS = {
    "deal-rs": "trxcc2002/DEAL-RS",
    "deal-ws": "trxcc2002/DEAL-WS",
}


def _extract_prefix(z: zipfile.ZipFile, prefix: str, destination: Path) -> int:
    prefix = prefix.rstrip("/") + "/"
    count = 0
    for info in z.infolist():
        if not info.filename.startswith(prefix):
            continue
        relative = PurePosixPath(info.filename[len(prefix):])
        if not relative.parts:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe archive path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Symlink entries are not permitted in archive: {info.filename}")
        target = destination.joinpath(*relative.parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    return count


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _resolve_hf_dataset_dir(benchmark: str, hf_local_dir: Path | None, cache_dir: Path) -> Path:
    if hf_local_dir is not None:
        resolved = hf_local_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"--hf-local-dir does not exist: {resolved}")
        return resolved
    from huggingface_hub import snapshot_download

    repo_id = HF_REPOS[benchmark]
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir / benchmark),
    )
    return Path(downloaded).resolve()


def _prepare_deal_rs(hf_dir: Path, out_dir: Path) -> dict:
    import pyarrow.parquet as pq

    zip_cache: dict[Path, zipfile.ZipFile] = {}

    def get_zip(rel_path: str) -> zipfile.ZipFile:
        full = (hf_dir / rel_path).resolve()
        if full not in zip_cache:
            zip_cache[full] = zipfile.ZipFile(full, "r")
        return zip_cache[full]

    counts: dict[str, int] = {}
    try:
        for hf_split, out_name in (("train", "train"), ("test", "eval")):
            rows = pq.read_table(hf_dir / "data" / f"{hf_split}.parquet").to_pylist()
            split_dir = out_dir / out_name
            split_dir.mkdir(parents=True, exist_ok=True)
            jsonl_lines: list[str] = []
            for idx, row in enumerate(rows):
                task_id = str(row["task_id"])
                task_dir = split_dir / task_id
                input_dir = task_dir / "input_files"
                reward_dir = task_dir / "reward"
                input_dir.mkdir(parents=True, exist_ok=True)
                reward_dir.mkdir(parents=True, exist_ok=True)

                ws_files = _extract_prefix(get_zip(row["workspace_archive"]), row["workspace_prefix"], input_dir)
                if ws_files == 0:
                    raise ValueError(f"No workspace files extracted for {task_id}")
                ev_files = _extract_prefix(get_zip(row["evaluation_archive"]), row["evaluation_prefix"], reward_dir)
                if ev_files == 0:
                    raise ValueError(f"No evaluation files extracted for {task_id}")

                metadata = {
                    "source": "researchclaw",
                    "family_id": row.get("family_id") or row.get("family") or "",
                    "paper_ids": row.get("paper_ids") or [],
                    "repo_ids": row.get("repo_ids") or [],
                    "ordinal": idx,
                }
                entry = {
                    "task_id": task_id,
                    "user_query": str(row["prompt"]),
                    "input_mount_dir": "/root/.openclaw/workspace",
                    "metadata": metadata,
                }
                (task_dir / "data_entry.json").write_text(
                    json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                jsonl_lines.append(json.dumps(entry, ensure_ascii=False))
            (split_dir / "dataset.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
            counts[out_name] = len(rows)
    finally:
        for z in zip_cache.values():
            z.close()

    return {
        "benchmark": "deal-rs",
        "source_repo": HF_REPOS["deal-rs"],
        "train_dir": str((out_dir / "train").resolve()),
        "eval_dir": str((out_dir / "eval").resolve()),
        "train_count": counts["train"],
        "eval_count": counts["eval"],
        "prepared_at": time.time(),
    }


def _prepare_deal_ws(
    hf_dir: Path,
    out_dir: Path,
    workspace_zip: Path | None,
    *,
    download_workspace_zip: bool,
    verify_workspace_zip: bool,
) -> dict:
    import pyarrow.parquet as pq

    ref = json.loads((hf_dir / "workspace-reference.json").read_text(encoding="utf-8"))
    if workspace_zip is None:
        if not download_workspace_zip:
            raise ValueError(
                "DEAL-WS requires --workspace-zip /path/to/filesys_en.zip "
                "(or pass --download-workspace-zip to fetch from "
                f"{ref['repo_id']}@{ref['revision']})."
            )
        from huggingface_hub import hf_hub_download

        downloaded_zip = hf_hub_download(
            repo_id=ref["repo_id"],
            repo_type=ref.get("repo_type", "dataset"),
            revision=ref["revision"],
            filename=ref["path"],
        )
        workspace_zip = Path(downloaded_zip).resolve()
    else:
        workspace_zip = workspace_zip.expanduser().resolve()

    if not workspace_zip.is_file():
        raise FileNotFoundError(f"Workspace ZIP not found: {workspace_zip}")
    if workspace_zip.stat().st_size != int(ref["size"]):
        raise ValueError(
            f"Unexpected filesys_en.zip size: {workspace_zip.stat().st_size} != {ref['size']}"
        )
    if verify_workspace_zip:
        _verify_sha256(workspace_zip, ref["sha256"])

    # Extract each upstream persona directory once into out_dir/filesys/<Persona_Workdir>
    filesys_root = out_dir / "filesys"
    filesys_root.mkdir(parents=True, exist_ok=True)
    persona_dirs = sorted(set(ref["persona_roots"].values()))
    with zipfile.ZipFile(workspace_zip, "r") as wz:
        for pdir in persona_dirs:
            target = filesys_root / pdir
            marker = target / ".EXTRACTED_OK"
            if marker.is_file():
                continue
            target.mkdir(parents=True, exist_ok=True)
            n = _extract_prefix(wz, pdir, target)
            if n == 0:
                raise ValueError(f"Failed to extract persona workspace {pdir} from {workspace_zip}")
            marker.write_text(f"{n}\n", encoding="utf-8")

    zip_cache: dict[Path, zipfile.ZipFile] = {}

    def get_zip(rel_path: str) -> zipfile.ZipFile:
        full = (hf_dir / rel_path).resolve()
        if full not in zip_cache:
            zip_cache[full] = zipfile.ZipFile(full, "r")
        return zip_cache[full]

    eval_personas_path = Path(__file__).resolve().parents[1] / "configs" / "deal-ws-eval-personas.json"
    eval_personas = (
        json.loads(eval_personas_path.read_text(encoding="utf-8"))
        if eval_personas_path.is_file()
        else {}
    )
    counts: dict[str, int] = {}
    try:
        for hf_split, out_name in (("train", "train"), ("test", "eval")):
            rows = pq.read_table(hf_dir / "data" / f"{hf_split}.parquet").to_pylist()
            split_dir = out_dir / out_name
            tasks_dir = split_dir / "tasks"
            tasks_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(rows):
                task_dir = tasks_dir / f"task_{idx:04d}"
                input_dir = task_dir / "input_files"
                reward_dir = task_dir / "reward"
                input_dir.mkdir(parents=True, exist_ok=True)
                reward_dir.mkdir(parents=True, exist_ok=True)

                _extract_prefix(get_zip(row["support_archive"]), row["support_prefix"], input_dir)
                ev_files = _extract_prefix(get_zip(row["evaluation_archive"]), row["evaluation_prefix"], reward_dir)
                if ev_files == 0:
                    raise ValueError(f"No evaluation files extracted for {row['task_id']}")

                persona_root = (filesys_root / row["workspace_root"]).resolve()
                assert persona_root.is_dir(), f"Missing extracted persona workspace: {persona_root}"

                source_task_id = str(row.get("source_task_id") or row["task_id"])
                scenario_persona = str(
                    row.get("scenario_persona") or eval_personas.get(source_task_id) or ""
                ).strip() or None
                entry = {
                    "task_id": source_task_id,
                    "user_query": str(row["prompt"]),
                    "input_mount_dir": "/root/.openclaw/workspace",
                    "metadata": {
                        "source": "wsclaw",
                        "hf_task_id": str(row["task_id"]),
                        "persona": str(row["persona"]),
                        **({"scenario_persona": scenario_persona} if scenario_persona else {}),
                        "required_source_files": list(row["required_source_files"]),
                        "required_output_paths": list(row["required_output_paths"]),
                        "workspace_mode": "full",
                        "workspace_root": str(persona_root),
                        "input_materialization": "full_workspace_top_level_mount_links",
                    },
                }
                (task_dir / "data_entry.json").write_text(
                    json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                contract_path = reward_dir / "output_contract.json"
                contract_data = (
                    json.loads(contract_path.read_text(encoding="utf-8"))
                    if contract_path.is_file()
                    else {}
                )
                task_output = {
                    "task_id": source_task_id,
                    "workspace_persona": str(row["persona"]),
                    "scenario_persona": scenario_persona,
                    "task_instruction": str(row["prompt"]),
                    "output_files": contract_data.get("output_files") or [],
                }
                (task_dir / "task_output.json").write_text(
                    json.dumps(task_output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            counts[out_name] = len(rows)
    finally:
        for z in zip_cache.values():
            z.close()

    return {
        "benchmark": "deal-ws",
        "source_repo": HF_REPOS["deal-ws"],
        "workspace_reference": ref["repo_id"] + "@" + ref["revision"],
        "filesys_root": str(filesys_root.resolve()),
        "train_dir": str((out_dir / "train").resolve()),
        "eval_dir": str((out_dir / "eval").resolve()),
        "train_count": counts["train"],
        "eval_count": counts["eval"],
        "prepared_at": time.time(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=["deal-rs", "deal-ws"], required=True)
    ap.add_argument("--output", type=Path, help="Output directory (defaults to ./data/<benchmark>)")
    ap.add_argument("--hf-local-dir", type=Path, help="Use an already downloaded local copy of the HF dataset")
    ap.add_argument("--hf-cache-dir", type=Path, default=Path("./data/.hf-cache"))
    ap.add_argument("--workspace-zip", type=Path, help="DEAL-WS only: path to Workspace-Bench filesys_en.zip")
    ap.add_argument("--download-workspace-zip", action="store_true", help="DEAL-WS only: download filesys_en.zip from Hugging Face")
    ap.add_argument("--verify-workspace-zip", action="store_true", help="DEAL-WS only: verify SHA-256 of filesys_en.zip")
    args = ap.parse_args()

    out_dir = (args.output or Path("data") / args.benchmark).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = _resolve_hf_dataset_dir(args.benchmark, args.hf_local_dir, args.hf_cache_dir.expanduser().resolve())

    if args.benchmark == "deal-rs":
        manifest = _prepare_deal_rs(hf_dir, out_dir)
    else:
        manifest = _prepare_deal_ws(
            hf_dir,
            out_dir,
            args.workspace_zip,
            download_workspace_zip=args.download_workspace_zip,
            verify_workspace_zip=args.verify_workspace_zip,
        )

    ready_path = out_dir / "DATA_READY.json"
    ready_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
