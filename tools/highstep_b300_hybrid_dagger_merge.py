#!/usr/bin/env python3
"""Create one immutable aggregate for a preregistered B300 DAgger round."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


WORKFLOW = "highstep_b300_canonical_hybrid_prior_latent_20260718"
FIELDS = (
    "student_obs_570",
    "critic_obs",
    "teacher_latent_raw_64",
    "teacher_latent_clamped_64",
    "teacher_pre_prior_action_16",
    "teacher_post_prior_policy_action_16",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_manifest(path: Path, expected_sha: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if sha256(path) != expected_sha:
        raise RuntimeError(f"manifest SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-amendment", required=True)
    parser.add_argument("--route-amendment-sha256", required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--prior-aggregate-manifest")
    parser.add_argument("--prior-aggregate-manifest-sha256")
    parser.add_argument("--dagger-manifest", action="append", default=[])
    parser.add_argument("--dagger-manifest-sha256", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    route_path = Path(args.route_amendment)
    route = load_manifest(route_path, args.route_amendment_sha256)
    if not (
        route.get("kind") == "highstep_b300_hybrid_dagger_route_amendment"
        and route.get("workflow_id") == WORKFLOW
        and route.get("status")
        == "frozen_before_D1_collection_after_E300_diagnosis"
        and route.get("maximum_rounds") == 3
        and route.get("training_updates_per_round") == 100
    ):
        raise RuntimeError("DAgger route authority mismatch")
    trajectories_per_round = int(route.get("collection_trajectories_per_round", -1))
    if (
        trajectories_per_round != len(route.get("scenarios", []))
        or len(args.dagger_manifest) != trajectories_per_round
        or len(args.dagger_manifest_sha256) != trajectories_per_round
    ):
        raise RuntimeError("DAgger trajectory count does not match frozen route")

    sources: list[tuple[Path, dict[str, Any]]] = []
    if args.round == 1:
        canonical_path = Path(args.canonical_manifest)
        canonical = load_manifest(canonical_path, args.canonical_manifest_sha256)
        if not (
            canonical.get("kind") == "highstep_b300_canonical_hybrid_tensor_trajectory"
            and canonical.get("unique_episode_count") == 1
            and canonical.get("sample_count") == 138
        ):
            raise RuntimeError("canonical dataset contract mismatch")
        sources.append((canonical_path, canonical))
    else:
        if not args.prior_aggregate_manifest or not args.prior_aggregate_manifest_sha256:
            raise RuntimeError("rounds two and three require the previous immutable aggregate")
        prior_path = Path(args.prior_aggregate_manifest)
        prior = load_manifest(prior_path, args.prior_aggregate_manifest_sha256)
        if not (
            prior.get("kind") == "highstep_b300_hybrid_dagger_aggregate"
            and prior.get("workflow_id") == WORKFLOW
            and prior.get("round") == args.round - 1
        ):
            raise RuntimeError("previous DAgger aggregate contract mismatch")
        sources.append((prior_path, prior))

    seen_scenarios: set[str] = set()
    for name, expected_sha in zip(args.dagger_manifest, args.dagger_manifest_sha256, strict=True):
        path = Path(name)
        manifest = load_manifest(path, expected_sha)
        if not (
            manifest.get("kind") == "highstep_b300_hybrid_dagger_tensor_trajectory"
            and manifest.get("workflow_id") == WORKFLOW
            and manifest.get("round") == args.round
            and manifest.get("student_drives_physics") is True
            and manifest.get("teacher_labels_same_pre_step_state") is True
            and manifest.get("sample_count") == 138
        ):
            raise RuntimeError(f"DAgger trajectory contract mismatch: {path}")
        scenario_payload = manifest.get("scenario")
        scenario = (
            str(scenario_payload.get("name", ""))
            if isinstance(scenario_payload, dict)
            else str(scenario_payload)
        )
        if scenario in seen_scenarios:
            raise RuntimeError(f"duplicate DAgger scenario: {scenario}")
        seen_scenarios.add(scenario)
        sources.append((path, manifest))
    expected_scenarios = {str(row["name"]) for row in route["scenarios"]}
    if seen_scenarios != expected_scenarios:
        raise RuntimeError("DAgger scenario matrix mismatch")

    payloads: list[dict[str, torch.Tensor]] = []
    source_evidence = []
    for manifest_path, manifest in sources:
        dataset_path = Path(manifest["dataset_path"]).resolve(strict=True)
        if sha256(dataset_path) != manifest["dataset_sha256"]:
            raise RuntimeError(f"dataset SHA mismatch: {dataset_path}")
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
        payloads.append(payload)
        source_evidence.append({
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path.resolve()),
            "dataset": str(dataset_path),
            "dataset_sha256": manifest["dataset_sha256"],
        })

    merged = {}
    for field in FIELDS:
        values = [payload.get(field) for payload in payloads]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError(f"aggregate source lacks {field}")
        merged[field] = torch.cat(values, dim=0).contiguous()
        if not bool(torch.all(torch.isfinite(merged[field])).item()):
            raise RuntimeError(f"aggregate contains non-finite {field}")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output}")
    output.mkdir(parents=True)
    dataset_path = output / "training_dataset.pt"
    torch.save(merged, dataset_path)
    unique_episodes = 1 + trajectories_per_round * args.round
    expected_samples = 138 * unique_episodes
    if merged["student_obs_570"].shape[0] != expected_samples:
        raise RuntimeError("aggregate sample count does not equal its unique episodes")
    manifest = {
        "schema_version": 1,
        "kind": "highstep_b300_hybrid_dagger_aggregate",
        "workflow_id": WORKFLOW,
        "status": "frozen_training_dataset",
        "round": args.round,
        "sample_count": expected_samples,
        "unique_episode_count": unique_episodes,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "route_amendment": str(route_path.resolve()),
        "route_amendment_sha256": args.route_amendment_sha256,
        "preregistration_sha256": route["preregistration_sha256"],
        "canonical_tensor_dataset_sha256": route["canonical_tensor_dataset_sha256"],
        "source_evidence": source_evidence,
        "fields": {name: list(value.shape) for name, value in merged.items()},
    }
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "sample_count": expected_samples,
        "unique_episode_count": unique_episodes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
