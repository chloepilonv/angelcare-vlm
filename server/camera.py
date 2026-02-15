"""
AngelCare — RTSP Camera Capture
=================================
Background loop that captures short clips from an RTSP camera stream
and feeds them into the AngelCare analysis pipeline.

Supports any IP camera with RTSP (Tapo TC200/C200, Reolink, Hikvision, etc).
Uses ffmpeg to grab clips — no OpenCV or GStreamer needed.

Usage:
    from server.camera import CameraCapture

    capture = CameraCapture(
        rtsp_url="rtsp://user:pass@192.168.1.100:554/stream1",
        on_clip=my_callback,   # called with path to each captured clip
        interval=30,           # seconds between captures
        clip_duration=10,      # seconds per clip
    )
    capture.start()   # runs in background thread
    capture.stop()
"""

import subprocess
import threading
import time
from pathlib import Path


CAPTURE_DIR = Path("captures")


class CameraCapture:
    """Background RTSP clip capture loop."""

    def __init__(
        self,
        rtsp_url: str,
        on_clip,
        interval: int = 30,
        clip_duration: int = 10,
    ):
        """
        Args:
            rtsp_url: RTSP stream URL (e.g. rtsp://user:pass@ip:554/stream1)
            on_clip: Callback function(clip_path: str) called after each capture
            interval: Seconds between the start of each capture
            clip_duration: Duration of each clip in seconds
        """
        self.rtsp_url = rtsp_url
        self.on_clip = on_clip
        self.interval = interval
        self.clip_duration = clip_duration
        self._thread = None
        self._stop_event = threading.Event()
        self._clip_count = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the capture loop in a background thread."""
        if self.running:
            return
        CAPTURE_DIR.mkdir(exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  Camera capture: started ({self.clip_duration}s clips every {self.interval}s)")
        print(f"  RTSP URL: {self._safe_url()}")

    def stop(self):
        """Stop the capture loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None
        print("  Camera capture: stopped")

    def _loop(self):
        """Main capture loop — runs in background thread."""
        while not self._stop_event.is_set():
            try:
                clip_path = self._capture_clip()
                if clip_path:
                    self.on_clip(clip_path)
            except Exception as e:
                print(f"[capture] Error: {e}")

            # Wait for next interval (interruptible)
            self._stop_event.wait(timeout=self.interval)

    def _capture_clip(self) -> str | None:
        """Capture a single clip from the RTSP stream using ffmpeg."""
        self._clip_count += 1
        timestamp = int(time.time())
        filename = f"cam_{timestamp}_{self._clip_count:04d}.mp4"
        output_path = CAPTURE_DIR / filename

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-t", str(self.clip_duration),
            "-c:v", "copy",        # no re-encoding for speed
            "-an",                  # drop audio (not needed for safety analysis)
            "-y",                   # overwrite if exists
            str(output_path),
            "-loglevel", "error",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.clip_duration + 15)

        if result.returncode != 0:
            print(f"[capture] ffmpeg error: {result.stderr.strip()}")
            return None

        if output_path.exists() and output_path.stat().st_size > 1000:
            return str(output_path)

        return None

    def _safe_url(self) -> str:
        """Return RTSP URL with password masked for logging."""
        import re
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", self.rtsp_url)

    def status(self) -> dict:
        """Return capture status for the API."""
        return {
            "running": self.running,
            "rtsp_url": self._safe_url() if self.rtsp_url else None,
            "clips_captured": self._clip_count,
            "interval": self.interval,
            "clip_duration": self.clip_duration,
        }
