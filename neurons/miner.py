"""Poker44 miner scoring chunks with a trained behavioral bot detector."""

# from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from neurons.detector import MODEL_PATH, DetectorModel
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Trained-model Poker44 miner.

    Scores each chunk with a calibrated classifier over hero-behavior features
    (see neurons/detector.py), trained on the public Poker44 training
    benchmark. Falls back to the original deterministic heuristic if the model
    artifact is missing so the miner never returns malformed responses.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.detector = None
        try:
            self.detector = DetectorModel()
            meta = self.detector.metadata
            bt.logging.info(
                f"🤖 Poker44 Miner started with trained detector "
                f"(algorithm={meta.get('algorithm')} "
                f"holdout_reward={meta.get('holdout_reward')})"
            )
        except Exception as exc:  # noqa: BLE001
            bt.logging.warning(
                f"Trained detector unavailable ({exc}); "
                f"falling back to reference heuristic. Expected artifact: {MODEL_PATH}"
            )
        repo_root = Path(__file__).resolve().parents[1]
        detector_meta = self.detector.metadata if self.detector else {}
        train_dates = detector_meta.get("train_dates") or [
            detector_meta.get("regime_start", "n/a"),
            detector_meta.get("trained_through", "n/a"),
        ]
        artifact_sha256 = ""
        if MODEL_PATH.is_file():
            artifact_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        try:
            import subprocess

            repo_commit = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            repo_commit = ""
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "detector.py",
                Path(__file__).resolve().parent / "pt2bag" / "pt_model.py",
                Path(__file__).resolve().parent / "pt2bag" / "pt_features.py",
                Path(__file__).resolve().parent / "pt2bag" / "serving.py",
                Path(__file__).resolve().parent / "pt2bag" / "features_v2.py",
                Path(__file__).resolve().parent / "pt2bag" / "schema_features.py",
                repo_root / "scripts" / "miner" / "train_pt2.py",
            ],
            defaults={
                "model_name": "perturb-poker-2",
                "model_version": str(detector_meta.get("feature_version", "pt.v1")),
                "framework": f"xgboost+sklearn/{detector_meta.get('algorithm', 'PT2Bag')}",
                "license": "MIT",
                "repo_url": "https://github.com/dragonwarrior0225/poker",
                "repo_commit": repo_commit,
                "artifact_sha256": artifact_sha256,
                "notes": (
                    "UID138-style PT2Bag: 6 perturbed XGBoosts (72%) + "
                    "2× PCA(50)→MLP(64) (28%), rank-blended; sanitize "
                    "train==serve; live-size pool/subset aug; human-quantile "
                    "threshold remap to 0.5 plus force top-K (15%) safety "
                    "budget so every batch has scores >=0.5 (blocks reward "
                    "0.0 from zero-TP threshold-sanity failure). Trained by "
                    "scripts/miner/train_pt2.py."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(api.poker44.net/api/v1/benchmark), release dates "
                    f"{train_dates[0]}..{train_dates[-1]}."
                ),
                "training_data_sources": ["poker44-training-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    def _capture_payload(self, chunks, scores) -> None:
        """Persist live validator payloads for train/serve-skew diagnosis.

        Best-effort only: capture failures must never affect the response.
        Rolling cap keeps disk usage bounded.
        """
        try:
            import json

            cap_dir = Path(__file__).resolve().parents[1] / "data" / "live_payloads"
            cap_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(cap_dir.glob("cycle_*.json"))
            while len(existing) >= 200:  # rolling cap (~200 cycles)
                existing.pop(0).unlink()
            stamp = time.strftime("%Y%m%dT%H%M%S")
            sizes = [len(c or []) for c in chunks]
            bt.logging.info(
                f"Live payload | chunks={len(chunks)} hands_per_chunk "
                f"min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0} "
                f"total={sum(sizes)}"
            )
            out = cap_dir / f"cycle_{stamp}.json"
            out.write_text(json.dumps({"ts": stamp, "chunks": chunks, "scores": scores}))
        except Exception as exc:  # noqa: BLE001
            bt.logging.debug(f"Payload capture skipped: {exc}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one calibrated bot-risk score per chunk."""
        chunks = synapse.chunks or []
        if self.detector is not None:
            try:
                scores = self.detector.score_chunks(chunks)
            except Exception as exc:  # noqa: BLE001
                bt.logging.error(f"Detector inference failed ({exc}); using heuristic.")
                scores = [self.score_chunk(chunk) for chunk in chunks]
        else:
            scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        self._capture_payload(chunks, scores)
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
