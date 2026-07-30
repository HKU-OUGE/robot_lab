#!/usr/bin/env python3
"""Freeze canonical plus accumulated raw-action DAgger episodes."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--dagger-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifests = [json.loads(args.canonical_manifest.read_text())]
    manifests += [json.loads(path.read_text()) for path in args.dagger_manifest]
    if manifests[0].get("passed") is not True: raise RuntimeError("canonical smoke is not passed")
    arrays = []
    for index, manifest in enumerate(manifests):
        path = Path(manifest["dataset_path"])
        if sha(path) != manifest["dataset_sha256"]: raise RuntimeError(f"dataset SHA mismatch: {path}")
        item = dict(np.load(path, allow_pickle=False)); item["episode_id"][:] = index; arrays.append(item)
    keys = tuple(arrays[0])
    if any(tuple(item) != keys for item in arrays): raise RuntimeError("DAgger dataset schema changed")
    merged = {key: np.concatenate([item[key] for item in arrays], axis=0) for key in keys}
    dataset = args.output_dir / "raw_action_aggregate.npz"; np.savez_compressed(dataset, **merged)
    result = {"schema_version": 1, "kind": "highstep_fixed_motion_raw_action_aggregate",
              "status": "passed", "passed": True,
              "workflow_id": "highstep_fixed_motion_raw_action_20260717",
              "dataset_path": str(dataset.resolve()), "dataset_sha256": sha(dataset),
              "unique_episode_count": len(arrays), "sample_count": int(merged["step"].size),
              "preregistration_sha256": manifests[0]["preregistration_sha256"],
              "source_manifests": [str(args.canonical_manifest.resolve())] + [str(p.resolve()) for p in args.dagger_manifest]}
    manifest = args.output_dir / "aggregate_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest), "sha256": sha(manifest)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
