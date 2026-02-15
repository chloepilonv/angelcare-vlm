"""
AngelCare — Speculative Cascade Inference
==========================================
Implements a speculative cascade (ICLR 2025) for video safety analysis.

    Cosmos Reason 2 (8B)  →  deferral check  →  Nemotron VL (12B)
         fast drafter            gate              verifier

The fast model (Cosmos) runs on every clip.  A deferral rule inspects
the result for uncertainty signals.  When triggered, the verifier
(Nemotron) re-analyzes the same video and its result is used instead.

Deferral signals (any one triggers escalation):
    • Risk level is MEDIUM (ambiguous — could be false alarm or missed danger)
    • JSON parse failed (prediction_class_id == -1)
    • Video description is very short (< 30 chars) or empty
    • Risk event flagged but no temporal_segment provided
    • Risk level is UNKNOWN

Reference:
    "Faster Cascades via Speculative Decoding" — ICLR 2025
    https://arxiv.org/abs/2405.19261
"""

from dataclasses import dataclass, field


@dataclass
class CascadeStats:
    """Tracks cascade performance metrics."""
    total: int = 0
    deferred: int = 0
    cosmos_only: int = 0
    agreements: int = 0  # when both models agree on risk level

    @property
    def deferral_rate(self) -> float:
        return self.deferred / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Cascade: {self.total} clips | "
            f"{self.deferred} deferred ({self.deferral_rate:.0%}) | "
            f"{self.cosmos_only} Cosmos-only"
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "deferred": self.deferred,
            "cosmos_only": self.cosmos_only,
            "deferral_rate": round(self.deferral_rate, 3),
            "agreements": self.agreements,
        }


def should_defer(result: dict) -> tuple[bool, str]:
    """
    Decide whether a Cosmos result is uncertain and should be deferred
    to the Nemotron verifier.

    Args:
        result: Analysis result from Cosmos Reason 2

    Returns:
        Tuple of (should_defer, reason)
    """
    risk = result.get("risk_level", "UNKNOWN")
    desc = result.get("video_description", "")
    class_id = result.get("prediction_class_id", -1)
    temporal = result.get("risk_assessment", {}).get("temporal_segment")

    # Parse failure — model wasn't confident enough to produce structured output
    if class_id == -1:
        return True, "parse_error"

    # Unknown risk level
    if risk == "UNKNOWN":
        return True, "unknown_risk"

    # MEDIUM risk is inherently ambiguous
    if risk == "MEDIUM":
        return True, "medium_risk"

    # Very short description suggests low model confidence
    if len(desc) < 30:
        return True, "short_description"

    # Risk event detected but no temporal segment — model unsure about timing
    if risk in ("CRITICAL", "HIGH") and temporal is None:
        return True, "missing_temporal"

    return False, "confident"


class CascadeAnalyzer:
    """
    Speculative cascade: Cosmos (fast) → deferral gate → Nemotron (verifier).

    Both models must be pre-loaded and passed to the constructor.
    """

    def __init__(self, cosmos_model, cosmos_processor,
                 nemotron_model, nemotron_tokenizer, nemotron_processor):
        self.cosmos_model = cosmos_model
        self.cosmos_processor = cosmos_processor
        self.nemotron_model = nemotron_model
        self.nemotron_tokenizer = nemotron_tokenizer
        self.nemotron_processor = nemotron_processor
        self.stats = CascadeStats()

    def analyze(self, video_path: str) -> dict:
        """
        Run speculative cascade on a single video.

        1. Cosmos produces the draft result
        2. Deferral rule checks for uncertainty
        3. If uncertain, Nemotron re-analyzes the video

        Args:
            video_path: Path to the video file

        Returns:
            Analysis result dict with extra cascade metadata:
                model: "cosmos" or "nemotron"
                deferred: bool
                deferral_reason: str
                cosmos_result: original Cosmos output (when deferred)
        """
        from core.inference import analyze_video
        from core.nemotron import analyze_video_nemotron

        # Stage 1: Cosmos draft
        cosmos_result = analyze_video(
            self.cosmos_model, self.cosmos_processor, video_path
        )

        self.stats.total += 1
        defer, reason = should_defer(cosmos_result)

        if not defer:
            # Cosmos is confident — use its result directly
            self.stats.cosmos_only += 1
            cosmos_result["model"] = "cosmos"
            cosmos_result["deferred"] = False
            cosmos_result["deferral_reason"] = reason
            return cosmos_result

        # Stage 2: Nemotron verification
        self.stats.deferred += 1
        nemotron_result = analyze_video_nemotron(
            self.nemotron_model, self.nemotron_tokenizer,
            self.nemotron_processor, video_path
        )

        # Track agreement
        if cosmos_result.get("risk_level") == nemotron_result.get("risk_level"):
            self.stats.agreements += 1

        # Attach cascade metadata
        nemotron_result["model"] = "nemotron"
        nemotron_result["deferred"] = True
        nemotron_result["deferral_reason"] = reason
        nemotron_result["cosmos_result"] = {
            "risk_level": cosmos_result.get("risk_level"),
            "prediction_label": cosmos_result.get("prediction_label"),
            "video_description": cosmos_result.get("video_description"),
        }

        return nemotron_result
