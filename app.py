"""
AngelCare — Web Dashboard
==========================
Flask app to view videos and their model analysis side by side.
Includes VSS proxy routes for livestream monitoring and Twilio alerts.

This server can run in four modes:
1. With local model inference (loads Cosmos Reason 2 directly)
2. No-model mode (serves pre-computed results only)
3. VSS integration mode (proxies requests to VSS backend)
4. Cascade mode (Cosmos Reason 2 + Nemotron VL speculative cascade)

The server maintains a ring buffer of recent events for the livestream
dashboard and automatically triggers SMS alerts for high-risk events.

Usage:
    python app.py                    # load model + serve web UI
    python app.py --no-model         # serve UI only (use pre-computed results)
    python app.py --vss-url http://localhost:8100   # enable VSS integration
    python app.py --cascade          # speculative cascade (both models)
"""

# Fix mamba_ssm / transformers compatibility (GreedySearchDecoderOnlyOutput removed in newer transformers)
import transformers.generation
if not hasattr(transformers.generation, 'GreedySearchDecoderOnlyOutput'):
    transformers.generation.GreedySearchDecoderOnlyOutput = transformers.generation.GenerateDecoderOnlyOutput
if not hasattr(transformers.generation, 'SampleDecoderOnlyOutput'):
    transformers.generation.SampleDecoderOnlyOutput = transformers.generation.GenerateDecoderOnlyOutput

import argparse
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

RESULTS_FILE = Path("angelcare_results.json")
VIDEOS_DIR = Path("videos")
UPLOAD_DIR = Path("uploads")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vercel frontend

# Global model references (loaded on startup unless --no-model)
_model = None
_processor = None
_cascade = None  # CascadeAnalyzer instance (when --cascade is used)

# VSS integration
_vss_client = None
_alert_sender = None
_camera = None  # CameraCapture instance (started via /api/camera/start)
_recent_events = []  # ring buffer of recent VSS events for the dashboard
_events_lock = threading.Lock()
MAX_EVENTS = 100  # maximum number of events to retain in the ring buffer


def get_results() -> list[dict]:
    """Load results from JSON file."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(results: list[dict]) -> None:
    """Persist results to JSON file."""
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


# ── Original routes ────────────────────────────────────────


@app.route("/")
def index():
    results = get_results()
    return render_template("index.html", results=results, model_loaded=_model is not None)


@app.route("/videos/<path:filepath>")
def serve_video(filepath):
    """Serve video files from the videos/ directory."""
    return send_from_directory(VIDEOS_DIR, filepath)


@app.route("/uploads/<path:filepath>")
def serve_upload(filepath):
    """Serve uploaded video files."""
    return send_from_directory(UPLOAD_DIR, filepath)


@app.route("/captures/<path:filepath>")
def serve_capture(filepath):
    """Serve captured camera clips."""
    return send_from_directory("captures", filepath)


@app.route("/api/results")
def api_results():
    return jsonify(get_results())


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Upload a video and run inference (direct model or cascade)."""
    if _model is None and _cascade is None:
        return jsonify({"error": "Model not loaded. Start server without --no-model."}), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "Empty filename."}), 400

    # Save uploaded file
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / video.filename
    video.save(str(save_path))

    # Run inference — cascade if available, otherwise Cosmos only
    if _cascade:
        result = _cascade.analyze(str(save_path))
    else:
        from core.inference import analyze_video
        result = analyze_video(_model, _processor, str(save_path))
        result["model"] = "cosmos"
        result["deferred"] = False

    result["file"] = f"uploads/{video.filename}"
    result["source"] = "upload"

    # Append to results
    results = get_results()
    results.append(result)
    save_results(results)

    return jsonify(result)


@app.route("/api/cascade/stats")
def cascade_stats():
    """Return cascade performance metrics."""
    if _cascade is None:
        return jsonify({"enabled": False})
    return jsonify({"enabled": True, **_cascade.stats.to_dict()})


# ── VSS proxy routes ──────────────────────────────────────


@app.route("/api/vss/health")
def vss_health():
    """Check if VSS backend is reachable."""
    if _vss_client is None:
        return jsonify({"status": "disabled", "error": "VSS not configured. Use --vss-url."}), 503
    healthy = _vss_client.health()
    if healthy:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "error": "VSS not reachable"}), 502


@app.route("/api/vss/upload", methods=["POST"])
def vss_upload():
    """Upload a video to VSS for analysis."""
    if _vss_client is None:
        return jsonify({"error": "VSS not configured."}), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "Empty filename."}), 400

    # Save locally then upload to VSS
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / video.filename
    video.save(str(save_path))

    try:
        video_id = _vss_client.upload_video(str(save_path))
        return jsonify({"video_id": video_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/vss/summarize/<video_id>")
def vss_summarize(video_id):
    """Get VSS summary/captions for a video. Defers to Nemotron if cascade is enabled."""
    if _vss_client is None:
        return jsonify({"error": "VSS not configured."}), 503

    try:
        result = _vss_client.summarize_video(video_id)
        result["model"] = "cosmos"
        result["deferred"] = False

        # Cascade: check if VSS result should be deferred to Nemotron
        if _cascade:
            from core.cascade import should_defer
            defer, reason = should_defer(result)
            if defer:
                # Find the uploaded file for this video_id
                uploaded = list(UPLOAD_DIR.glob("*"))
                if uploaded:
                    video_path = str(max(uploaded, key=lambda p: p.stat().st_mtime))
                    from core.nemotron import analyze_video_nemotron
                    nemotron_result = analyze_video_nemotron(
                        _cascade.nemotron_model, _cascade.nemotron_tokenizer,
                        _cascade.nemotron_processor, video_path
                    )
                    nemotron_result["model"] = "nemotron"
                    nemotron_result["deferred"] = True
                    nemotron_result["deferral_reason"] = reason
                    nemotron_result["cosmos_result"] = {
                        "risk_level": result.get("risk_level"),
                        "summary": result.get("summary"),
                    }
                    _cascade.stats.total += 1
                    _cascade.stats.deferred += 1
                    result = nemotron_result
                else:
                    result["deferral_reason"] = f"would_defer:{reason}_no_file"
            else:
                _cascade.stats.total += 1
                _cascade.stats.cosmos_only += 1

        _push_event(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/vss/events")
def vss_events():
    """Return recent events for the livestream dashboard (polled by frontend)."""
    # Return events newer than the 'since' timestamp (epoch seconds)
    since = request.args.get("since", 0, type=float)
    with _events_lock:
        new_events = [e for e in _recent_events if e.get("timestamp", 0) > since]
    return jsonify({"events": new_events})


def _push_event(result: dict):
    """
    Add an event to the ring buffer and trigger alerts.

    This function is called when VSS produces a new analysis result.
    It maintains a thread-safe ring buffer for the livestream dashboard
    and automatically sends SMS alerts for CRITICAL/HIGH risk events.

    Args:
        result: VSS analysis result with summary and risk_level
    """
    result["timestamp"] = time.time()
    with _events_lock:
        _recent_events.insert(0, result)
        # Trim to max size (keeps most recent MAX_EVENTS)
        while len(_recent_events) > MAX_EVENTS:
            _recent_events.pop()
    # Send SMS if needed (rate-limited internally)
    if _alert_sender:
        _alert_sender.send_if_needed(result)


# ── Camera capture routes ─────────────────────────────────


def _on_camera_clip(clip_path: str):
    """Callback when a new clip is captured from the camera stream."""
    # Build clip URL so the dashboard can play the video
    clip_filename = Path(clip_path).name
    clip_url = f"/captures/{clip_filename}"

    if _vss_client:
        # VSS path: upload clip → summarize → push event
        try:
            video_id = _vss_client.upload_video(clip_path)
            result = _vss_client.summarize_video(video_id)
            result["model"] = "cosmos"
            result["deferred"] = False
            result["source"] = "camera"
            result["clip_url"] = clip_url

            # Cascade deferral on VSS result
            if _cascade:
                from core.cascade import should_defer
                defer, reason = should_defer(result)
                if defer:
                    from core.nemotron import analyze_video_nemotron
                    nemotron_result = analyze_video_nemotron(
                        _cascade.nemotron_model, _cascade.nemotron_tokenizer,
                        _cascade.nemotron_processor, clip_path
                    )
                    nemotron_result["model"] = "nemotron"
                    nemotron_result["deferred"] = True
                    nemotron_result["deferral_reason"] = reason
                    nemotron_result["source"] = "camera"
                    nemotron_result["clip_url"] = clip_url
                    _cascade.stats.total += 1
                    _cascade.stats.deferred += 1
                    result = nemotron_result
                else:
                    _cascade.stats.total += 1
                    _cascade.stats.cosmos_only += 1

            _push_event(result)
        except Exception as e:
            print(f"[camera] VSS analysis failed: {e}")

    elif _cascade:
        # Local cascade path (no VSS)
        try:
            result = _cascade.analyze(clip_path)
            result["source"] = "camera"
            result["clip_url"] = clip_url
            _push_event(result)
        except Exception as e:
            print(f"[camera] Cascade analysis failed: {e}")

    elif _model:
        # Local Cosmos-only path
        try:
            from core.inference import analyze_video
            result = analyze_video(_model, _processor, clip_path)
            result["model"] = "cosmos"
            result["deferred"] = False
            result["source"] = "camera"
            result["clip_url"] = clip_url
            _push_event(result)
        except Exception as e:
            print(f"[camera] Analysis failed: {e}")


@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    """Start capturing from an RTSP camera stream."""
    global _camera

    if _model is None and _cascade is None and _vss_client is None:
        return jsonify({"error": "No model or VSS configured. Cannot analyze clips."}), 503

    data = request.get_json() or {}
    rtsp_url = data.get("rtsp_url", "").strip()
    if not rtsp_url:
        return jsonify({"error": "rtsp_url is required."}), 400

    interval = data.get("interval", 30)
    clip_duration = data.get("clip_duration", 10)

    # Stop existing capture if running
    if _camera and _camera.running:
        _camera.stop()

    from server.camera import CameraCapture
    _camera = CameraCapture(
        rtsp_url=rtsp_url,
        on_clip=_on_camera_clip,
        interval=interval,
        clip_duration=clip_duration,
    )
    _camera.start()
    return jsonify({"status": "started", **_camera.status()})


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    """Stop the camera capture loop."""
    global _camera
    if _camera and _camera.running:
        _camera.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


@app.route("/api/camera/status")
def camera_status():
    """Return camera capture status."""
    if _camera:
        return jsonify(_camera.status())
    return jsonify({"running": False})


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for AngelCare Flask web server."""
    global _model, _processor, _cascade, _vss_client, _alert_sender

    parser = argparse.ArgumentParser(description="AngelCare Web Dashboard")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip model loading (view pre-computed results only)")
    parser.add_argument("--cascade", action="store_true",
                        help="Enable speculative cascade: Cosmos (8B) + Nemotron VL (12B)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--vss-url", type=str, default=None,
                        help="VSS API base URL (e.g. http://localhost:8100)")
    args = parser.parse_args()

    if args.cascade:
        from core.inference import load_model
        from core.cascade import CascadeAnalyzer
        from core.nemotron import load_nemotron

        _model, _processor = load_model()
        nem_model, nem_tokenizer, nem_processor = load_nemotron()
        _cascade = CascadeAnalyzer(_model, _processor, nem_model, nem_tokenizer, nem_processor)
        print("  Cascade mode: Cosmos Reason 2 → Nemotron VL")
    elif not args.no_model:
        from core.inference import load_model
        _model, _processor = load_model()
    else:
        print("Running without model — upload/analyze disabled.")

    if args.vss_url:
        from server.vss import VSSClient
        _vss_client = VSSClient(args.vss_url)
        print(f"  VSS integration: {args.vss_url}")
        if _vss_client.health():
            print("  VSS status: connected")
        else:
            print("  VSS status: not reachable (will retry)")

    # Initialize Twilio alerts (no-op if env vars not set)
    from server.alerts import AlertSender
    _alert_sender = AlertSender()
    if _alert_sender.enabled:
        print(f"  SMS alerts: enabled → {_alert_sender.to_number}")
    else:
        print("  SMS alerts: disabled (set TWILIO_* env vars to enable)")

    print(f"\n  AngelCare Dashboard: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
