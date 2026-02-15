"""
AngelCare — VSS API Client
============================
Thin wrapper around the NVIDIA Video Search & Summarization REST API.

This client provides a simple interface to the VSS service for uploading
video files and running elder-safety analysis using Cosmos Reason 2.
It automatically configures prompts for elder safety monitoring and
extracts risk levels from the free-text summaries.

Usage:
    from server.vss import VSSClient

    client = VSSClient("http://localhost:8100")
    video_id = client.upload_video("clip.mp4")
    summary = client.summarize_video(video_id)
"""

import re
import requests

# Elder-safety captioning prompt (matches core/inference.py classification)
CAPTION_PROMPT = (
    "You are an Elder Care Safety Monitor. Describe what the elderly person is doing "
    "in 1-2 sentences. Focus on their posture, movement, and behavior. "
    "End with a risk assessment: CRITICAL (fall, on ground), HIGH (immobile, distress), "
    "MEDIUM (unsteady), or SAFE (normal activity)."
)

SUMMARY_PROMPT = (
    "Combine these captions into a concise safety summary. "
    "State the overall risk level (CRITICAL / HIGH / MEDIUM / SAFE) and what happened."
)

# Keywords used to infer risk level from free-text VSS captions
_RISK_PATTERNS = {
    "CRITICAL": re.compile(r"\b(fall|fallen|lying on.*(floor|ground)|collapsed|unconscious)\b", re.I),
    "HIGH": re.compile(r"\b(immobile|not moving|unresponsive|distress|pain|stuck|clutching)\b", re.I),
    "MEDIUM": re.compile(r"\b(unsteady|stumbl|losing balance|wobbl|unstable)\b", re.I),
}


class VSSClient:
    """Client for the NVIDIA VSS REST API."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Upload ──────────────────────────────────────────────

    def upload_video(self, filepath: str) -> str:
        """Upload a video file to VSS. Returns the video ID."""
        with open(filepath, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/files",
                data={"purpose": "vision", "media_type": "video"},
                files={"file": ("video_file", f)},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        return resp.json()["id"]

    # ── Summarize ───────────────────────────────────────────

    def summarize_video(
        self,
        video_id: str,
        caption_prompt: str | None = None,
        chunk_duration: int = 10,
    ) -> dict:
        """
        Run captioning + summarization on an uploaded video.
        Returns {"summary": str, "risk_level": str, "raw": dict}.
        """
        body = {
            "id": video_id,
            "prompt": caption_prompt or CAPTION_PROMPT,
            "caption_summarization_prompt": SUMMARY_PROMPT,
            "summary_aggregation_prompt": SUMMARY_PROMPT,
            "model": "cosmos-reason2",
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.3,
            "chunk_duration": chunk_duration,
        }
        resp = requests.post(
            f"{self.base_url}/summarize",
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        summary = data["choices"][0]["message"]["content"]
        risk_level = self._infer_risk(summary)
        return {"summary": summary, "risk_level": risk_level, "raw": data}

    # ── Health ──────────────────────────────────────────────

    def health(self) -> bool:
        """Check if the VSS API is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.ok
        except requests.ConnectionError:
            return False

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _infer_risk(text: str) -> str:
        """
        Extract risk level from free-text summary.

        VSS returns captions as natural language rather than structured JSON.
        This method extracts risk level by:
        1. Looking for explicit risk level keywords (CRITICAL, HIGH, MEDIUM, SAFE)
        2. Falling back to pattern matching for risk-indicating phrases
        3. Defaulting to SAFE if no risk indicators found

        Args:
            text: Free-text summary from VSS

        Returns:
            Risk level string: "CRITICAL", "HIGH", "MEDIUM", or "SAFE"
        """
        # Check for explicit risk labels first
        upper = text.upper()
        for level in ("CRITICAL", "HIGH", "MEDIUM", "SAFE"):
            if level in upper:
                return level
        # Fall back to keyword matching for common risk phrases
        for level, pattern in _RISK_PATTERNS.items():
            if pattern.search(text):
                return level
        return "SAFE"
