#!/usr/bin/env python3
"""Minimal Streamlit UI for embedding + detecting audio watermarks."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
EMBED_SCRIPT = BASE_DIR / "embed.py"
DETECT_SCRIPT = BASE_DIR / "detect.py"
LOGS_DIR = BASE_DIR / "Logs"
EMBED_LOG = LOGS_DIR / "wm_singlepath_log.csv"
DETECT_LOG = LOGS_DIR / "wm_detect_log.csv"
RUNS_DIR = LOGS_DIR / "streamlit_runs"
EMBED_RUNS_DIR = RUNS_DIR / "embed"
DETECT_RUNS_DIR = RUNS_DIR / "detect"
for path in (LOGS_DIR, RUNS_DIR, EMBED_RUNS_DIR, DETECT_RUNS_DIR):
    path.mkdir(parents=True, exist_ok=True)
DEFAULT_SECRET = "demo-secret"
WATERMARKED_NAME = "watermarked.wav"
ENCODE_PARAMS = {
    "--seg": "0.20",
    "--repeat": "1",
    "--sync-every": "8",
    "--pilot-bits": "16",
}
DECODE_PARAMS = {
    "--seg": "0.20",
    "--repeat": "1",
    "--sync-every": "8",
    "--pilot-bits": "16",
    "--fec": "none",
}


def run_cli(cmd: List[str]) -> Dict[str, str | int]:
    """Execute a CLI command and capture stdout/stderr."""
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join(cmd),
    }


def make_run_dir(root: Path, prefix: str) -> Path:
    """Create a persistent run directory inside Logs for artifacts."""
    for _ in range(5):
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{prefix}_{stamp}_{uuid4().hex[:6]}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a run directory inside Logs.")


def embed_audio(uploaded_file, secret: str) -> Dict[str, str | int]:
    """Persist the upload, call embed.py with safe defaults, and return CLI output."""
    if not EMBED_SCRIPT.exists():
        raise FileNotFoundError("embed.py not found in project root.")
    run_dir = make_run_dir(EMBED_RUNS_DIR, "embed")
    input_name = Path(uploaded_file.name or "input.wav").name
    input_path = run_dir / input_name
    input_path.write_bytes(uploaded_file.getbuffer())
    output_path = run_dir / WATERMARKED_NAME

    cmd: List[str] = [
        sys.executable,
        str(EMBED_SCRIPT),
        "--mode",
        "encode",
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--secret",
        secret,
        "--key-id",
        "1",
        "--engine",
        "audioseal",
        "--device",
        "cpu",
        "--log",
        str(EMBED_LOG),
    ]
    for flag, value in ENCODE_PARAMS.items():
        cmd += [flag, value]

    result = run_cli(cmd)
    result["artifact_dir"] = str(run_dir)
    if result["returncode"] == 0 and output_path.exists():
        result["audio_bytes"] = output_path.read_bytes()
        result["output_path"] = str(output_path)
    return result


def detect_audio(uploaded_file, secret: str, nonce_hex: str = "") -> Dict[str, str | int]:
    """Persist the upload, call detect.py with safe defaults, and return CLI output."""
    if not DETECT_SCRIPT.exists():
        raise FileNotFoundError("detect.py not found in project root.")
    run_dir = make_run_dir(DETECT_RUNS_DIR, "detect")
    input_name = Path(uploaded_file.name or "watermarked.wav").name
    input_path = run_dir / input_name
    input_path.write_bytes(uploaded_file.getbuffer())
    result_json = run_dir / (Path(input_name).stem + ".wm.json")

    cmd: List[str] = [
        sys.executable,
        str(DETECT_SCRIPT),
        "--in",
        str(input_path),
        "--secret",
        secret,
        "--key-id",
        "1",
        "--device",
        "cpu",
        "--json",
        str(result_json),
        "--log-csv",
        str(DETECT_LOG),
        "--embed-log",
        str(EMBED_LOG),
        "--pull-embed-params",
    ]
    if nonce_hex:
        cmd += ["--nonce-hex", nonce_hex]
    for flag, value in DECODE_PARAMS.items():
        cmd += [flag, value]

    result = run_cli(cmd)
    result["artifact_dir"] = str(run_dir)
    result["output_json"] = str(result_json)
    if result["returncode"] == 0 and result_json.exists():
        result["result_json"] = result_json.read_text(encoding="utf-8")
    return result


def render_command_output(result: Dict[str, str | int]):
    """Show stdout/stderr in the UI for transparency."""
    st.caption(f"$ {result.get('command', '')}")
    st.code(result.get("stdout") or "(no stdout)", language="bash")
    if result.get("stderr"):
        st.code(result["stderr"], language="bash")


def main():
    st.set_page_config(page_title="Audio Watermark Toolkit", layout="wide")
    st.title("Audio Watermarking Toolkit")
    st.write("Upload audio, click a button, and download results. Defaults mirror the CLI defaults.")

    embed_tab, detect_tab = st.tabs(["Embed", "Detect"])

    with embed_tab:
        st.subheader("1. Upload audio to embed a watermark")
        embed_secret = st.text_input(
            "Secret / passphrase (use the same value for detection later)",
            value=DEFAULT_SECRET,
            type="password",
        )
        embed_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "flac", "aiff", "aif", "mp3", "ogg"],
            key="embed_file",
        )
        embed_btn = st.button("Generate watermarked audio", disabled=not embed_file)
        if embed_btn and embed_file:
            with st.spinner("Embedding watermark..."):
                try:
                    result = embed_audio(embed_file, embed_secret.strip() or DEFAULT_SECRET)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    render_command_output(result)
                    if result.get("returncode") == 0 and result.get("audio_bytes"):
                        st.success("Watermark embedded successfully.")
                        st.download_button(
                            "Download watermarked audio",
                            data=result["audio_bytes"],
                            file_name=WATERMARKED_NAME,
                            mime="audio/wav",
                        )
                        if EMBED_LOG.exists():
                            st.caption(f"Run logged to {EMBED_LOG.relative_to(BASE_DIR)}")
                    else:
                        st.error("Embedding failed. Check the logs above.")

    with detect_tab:
        st.subheader("2. Upload a watermarked file to detect + view metadata")
        detect_secret = st.text_input(
            "Secret / passphrase (must match the one used at embed time)",
            value=DEFAULT_SECRET,
            type="password",
            key="detect_secret",
        )
        detect_nonce = st.text_input(
            "Nonce hex (optional, copy from embed log if available)",
            value="",
            help="Needed if embeddings used randomized nonce scheduling.",
            key="detect_nonce",
        )
        detect_file = st.file_uploader(
            "Choose a watermarked audio file",
            type=["wav", "flac", "aiff", "aif", "mp3", "ogg"],
            key="detect_file",
        )
        detect_btn = st.button("Detect watermark", disabled=not detect_file)
        if detect_btn and detect_file:
            with st.spinner("Detecting watermark..."):
                try:
                    result = detect_audio(
                        detect_file,
                        detect_secret.strip() or DEFAULT_SECRET,
                        detect_nonce.strip(),
                    )
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    render_command_output(result)
                    if result.get("returncode") == 0 and result.get("result_json"):
                        st.success("Watermark detected.")
                        st.json(json.loads(result["result_json"]))
                        st.download_button(
                            "Download metadata JSON",
                            data=result["result_json"].encode("utf-8"),
                            file_name="detection.json",
                            mime="application/json",
                        )
                    else:
                        st.error("Detection failed. Check the logs above.")


if __name__ == "__main__":
    main()
