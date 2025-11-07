#!/usr/bin/env python3
"""
wm_singlepath.py — Single‑path audio watermark embedder (hardened)

What’s new in this update
-------------------------
- **Authenticated payloads (optional)**: add --auth-key to include HMAC‑SHA256 (truncated 16B).
- **Per-copy nonce**: random 32‑bit nonce in headers; included in HMAC input.
- **Energy‑gated placement**: avoid near‑silence segments; place frames in higher‑RMS windows.
- **Jittered starts**: random intra‑segment offsets (seeded) to reduce regularity.
- **Ultra‑short fallback**: if even micro doesn’t fit, embed anchor‑only packet (128‑bit hash).
- **Tunable sync**: --sync-interval controls sync frame density.
- **More accurate capacity math** & clearer errors.

Design
------
- One path. Prefer high‑capacity engine 'wavmark' (stub), else robust AudioSeal streaming frames.
- Pipeline: canonicalize → zlib → FEC‑lite (repetition) → interleave + sync → energy‑aware placement.
- Quality guardrails: preserve SR/channels, RMS cap, peak‑normalize.
- Forensics: versioned headers, key_id/key seeding, nonce, optional HMAC.

Decoding not included in this file.
"""

import argparse
import binascii
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import struct
import sys
import zlib
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

# --------- Engines & constants ---------
ENGINE_WAVMARK = "wavmark"
ENGINE_AUDIOSEAL = "audioseal"

AS_MODEL_NAME = "audioseal_wm_16bits"
AS_SR = 16000
MIN_SEG_SEC = 0.20

HEADER_VER = 3  # bumped due to header changes

# Sync marker (16-bit)
DEFAULT_SYNC_WORD = 0xB39F

# --------- I/O helpers ---------

def read_audio(path: str) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True)
    x = x.astype(np.float32, copy=False)
    return x, sr

def write_audio(path: str, x: np.ndarray, sr: int, pcm16: bool):
    subtype = "PCM_16" if pcm16 else None
    sf.write(path, x, sr, subtype=subtype)

def ensure_len(t: torch.Tensor, n: int) -> torch.Tensor:
    cur = t.numel()
    if cur == n:
        return t
    if cur > n:
        return t[:n]
    out = torch.zeros(n, dtype=t.dtype, device=t.device)
    out[:cur] = t
    return out

def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.clamp(torch.mean(x**2), min=1e-12)))

# --------- Bit/packet helpers ---------

def bytes_to_bits(b: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in b for i in range(7, -1, -1)]

def bits_to_frames(bits: List[int], width: int = 16) -> List[List[int]]:
    frames = []
    for i in range(0, len(bits), width):
        chunk = bits[i:i+width]
        if len(chunk) < width:
            chunk += [0] * (width - len(chunk))
        frames.append(chunk)
    return frames

def u16_to_bits(val: int) -> List[int]:
    return [ (val >> i) & 1 for i in range(15, -1, -1) ]

def insert_sync_frames(frames: List[List[int]], interval: int, sync_word: int) -> List[List[int]]:
    if interval <= 0:
        return frames[:]
    out = []
    cnt = 0
    sync_bits = u16_to_bits(sync_word & 0xFFFF)
    for fr in frames:
        if cnt % interval == 0:
            out.append(sync_bits)
        out.append(fr)
        cnt += 1
    out.append(sync_bits)  # trailing sync
    return out

# --------- Manifest helpers ---------

def canonical_json_bytes(md: dict) -> bytes:
    return json.dumps(md, sort_keys=True, separators=(",", ":")).encode("utf-8")

def compress_zlib(b: bytes) -> bytes:
    return zlib.compress(b, level=9)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_128(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()[:16]

def hmac16(key_bytes: bytes, data: bytes) -> bytes:
    if not key_bytes:
        return b"\x00"*16
    return hashlib.pbkdf2_hmac("sha256", data, key_bytes, iterations=1, dklen=16)

def short_code_u16(s: str) -> int:
    if s is None:
        return 0
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return struct.unpack(">H", d[:2])[0]

def auto_manifest(infile: str, published_at: str | None, model: str | None,
                  title: str | None, issuer: str | None, license_str: str | None,
                  text_path: str | None) -> dict:
    st = os.stat(infile)
    created = dt.datetime.utcfromtimestamp(st.st_mtime).replace(tzinfo=dt.timezone.utc)
    pub_iso = published_at or dt.datetime.now(dt.timezone.utc).isoformat()

    with open(infile, "rb") as f:
        sha256_in = sha256_hex(f.read())

    text_hash = None
    if text_path and os.path.exists(text_path):
        with open(text_path, "rb") as f:
            text_hash = sha256_hex(f.read())

    return {
        "schema": "ai.audio.manifest",
        "schema_ver": "1.0",
        "title": title or None,
        "issuer": issuer or None,
        "license": license_str or None,
        "created_at": created.isoformat(),
        "published_at": pub_iso,
        "generator_model": model or "unknown",
        "generator_engine": "TTS",
        "provenance": {"input_file_sha256": sha256_in},
        "text_hash_sha256": text_hash,
        "notes": None,
    }

# --------- Packet formats ---------
# Header changes (v3):
# magic(2) 'WM', ver(1), key_id(1), type(1), flags(1), nonce(4), len(2), hmac16(16), crc32(4), body(...)

TYPE_MICRO = 0
TYPE_FULL = 2
TYPE_ANCHOR = 3

def make_nonce(seed_bytes: bytes) -> int:
    h = hashlib.sha256(seed_bytes + os.urandom(16)).digest()
    return struct.unpack(">I", h[:4])[0]

def pack_full_payload(manifest: dict, key_id: int, auth_key: bytes, seed: bytes) -> bytes:
    body = compress_zlib(canonical_json_bytes(manifest))
    nonce = make_nonce(seed)
    magic = b"WM"
    ver = bytes([HEADER_VER & 0xFF])
    kid = bytes([key_id & 0xFF])
    typ = bytes([TYPE_FULL])
    flags = bytes([0])
    ln = len(body).to_bytes(2, "big")
    hmac_bytes = hmac16(auth_key, ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + body)
    crc = binascii.crc32(body).to_bytes(4, "big")
    head = magic + ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + hmac_bytes + crc
    return head + body

def pack_micro_payload(full_manifest_json: dict, published_at: str | None,
                       model: str | None, issuer: str | None,
                       key_id: int, auth_key: bytes, seed: bytes) -> bytes:
    nonce = make_nonce(seed)
    # micro body: epoch(4), model_code(2), issuer_code(2), flags(1), anchor(16)
    if published_at:
        try:
            epoch = int(dt.datetime.fromisoformat(published_at.replace("Z","+00:00")).timestamp())
        except Exception:
            epoch = int(dt.datetime.now(dt.timezone.utc).timestamp())
    else:
        epoch = int(dt.datetime.now(dt.timezone.utc).timestamp())
    model_code = short_code_u16(model or "unknown")
    issuer_code = short_code_u16(issuer or "unknown")
    anchor = sha256_128(canonical_json_bytes(full_manifest_json))
    body = struct.pack(">IHHB", epoch, model_code, issuer_code, 0) + anchor
    magic = b"WM"
    ver = bytes([HEADER_VER & 0xFF])
    kid = bytes([key_id & 0xFF])
    typ = bytes([TYPE_MICRO])
    flags = bytes([0])
    ln = len(body).to_bytes(2, "big")
    hmac_bytes = hmac16(auth_key, ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + body)
    crc = binascii.crc32(body).to_bytes(4, "big")
    head = magic + ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + hmac_bytes + crc
    return head + body

def pack_anchor_payload(full_manifest_json: dict, key_id: int, auth_key: bytes, seed: bytes) -> bytes:
    nonce = make_nonce(seed)
    anchor = sha256_128(canonical_json_bytes(full_manifest_json))
    body = anchor
    magic = b"WM"
    ver = bytes([HEADER_VER & 0xFF])
    kid = bytes([key_id & 0xFF])
    typ = bytes([TYPE_ANCHOR])
    flags = bytes([0])
    ln = len(body).to_bytes(2, "big")
    hmac_bytes = hmac16(auth_key, ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + body)
    crc = binascii.crc32(body).to_bytes(4, "big")
    head = magic + ver + kid + typ + flags + nonce.to_bytes(4, "big") + ln + hmac_bytes + crc
    return head + body

# --------- Scheduling & placement ---------

def energy_gated_segment_indices(mono_model: torch.Tensor, seg_len_m: int, gate_db: float) -> List[int]:
    n = mono_model.numel()
    n_segments = max(1, n // seg_len_m)
    if n_segments <= 1:
        return [0]
    # compute rms per segment
    rms_vals = []
    for i in range(n_segments):
        s = i * seg_len_m
        e = min(s + seg_len_m, n)
        seg = mono_model[s:e]
        rms_vals.append(rms(seg))
    rms_vals = np.array(rms_vals, dtype=np.float32)
    median = float(np.median(rms_vals)) + 1e-9
    db = 20.0 * np.log10(np.maximum(rms_vals, 1e-9) / median)
    keep = [i for i, v in enumerate(db) if v >= gate_db]  # e.g., gate_db=-20 keeps segments within 20 dB of median
    if not keep:
        keep = list(range(n_segments))
    return keep

def schedule_positions(pool_indices: List[int], total_needed: int, seed_bytes: bytes) -> List[int]:
    rnd = random.Random(sha256_hex(seed_bytes))
    idxs = pool_indices[:]
    rnd.shuffle(idxs)
    out = []
    pos = 0
    while len(out) < total_needed:
        if pos >= len(idxs):
            rnd.shuffle(idxs)
            pos = 0
        out.append(idxs[pos])
        pos += 1
    return out

# --------- Engines ---------

def embed_frames_audioseal(audio_np: np.ndarray, orig_sr: int, frames: List[List[int]],
                           seg_sec: float, repeat: int, gain: float,
                           rms_cap: float, key_seed: bytes, sync_interval: int, sync_word: int,
                           energy_gate_db: float, jitter_s: float) -> np.ndarray:
    from audioseal import AudioSeal
    seg_sec = max(MIN_SEG_SEC, seg_sec)

    n_samples, n_ch = audio_np.shape
    dur = n_samples / float(orig_sr)

    # analysis mono @ 16k
    mono = torch.from_numpy(audio_np.mean(axis=1)).float()
    to_model = Resample(orig_sr=orig_sr, new_freq=AS_SR)
    mono_model = to_model(mono.unsqueeze(0)).squeeze(0)

    seg_len_m = int(seg_sec * AS_SR)
    n_segments = max(1, mono_model.numel() // seg_len_m)

    # Insert sync frames
    frames_with_sync = insert_sync_frames(frames, interval=sync_interval, sync_word=sync_word)

    total_needed = len(frames_with_sync) * max(1, repeat)

    # Energy‑gated pool of indices
    pool = energy_gated_segment_indices(mono_model, seg_len_m, gate_db=energy_gate_db)
    if len(pool) < total_needed:
        # fallback to all segments if insufficient
        pool = list(range(n_segments))

    if total_needed > len(pool):
        # last resort: shrink seg len to fit more segments (but not below MIN_SEG_SEC)
        seg_auto = max(MIN_SEG_SEC, dur / (math.ceil(total_needed) + 1))
        seg_len_m = int(seg_auto * AS_SR)
        n_segments = max(1, mono_model.numel() // seg_len_m)
        pool = list(range(n_segments))
        if total_needed > len(pool):
            raise SystemExit(f"Not enough capacity: need {total_needed} segments, have {len(pool)}. "
                             f"Increase duration, lower repeat, or lower seg.")

    positions = schedule_positions(pool, total_needed, seed_bytes=key_seed + struct.pack(">I", n_samples))

    wm_buf = torch.zeros_like(mono_model)
    gen = AudioSeal.load_generator(AS_MODEL_NAME)

    # jitter in samples at model SR
    jitter_s = max(0.0, float(jitter_s))
    jitter_m = int(jitter_s * AS_SR)
    rnd = random.Random(sha256_hex(key_seed + b"jitter"))

    idx = 0
    for fr in frames_with_sync:
        for r in range(repeat):
            seg_idx = positions[idx]; idx += 1
            base = seg_idx * seg_len_m
            # jitter start within segment
            j = rnd.randint(-jitter_m//2, jitter_m//2) if jitter_m > 0 else 0
            s = max(0, min(base + j, mono_model.numel() - seg_len_m))
            e = s + seg_len_m
            seg = mono_model[s:e].contiguous()
            message = torch.tensor([[int(b) for b in fr]], dtype=torch.int32)
            wm_seg = gen.get_watermark(seg, message, sample_rate=AS_SR)
            wm_buf[s:e] = wm_buf[s:e] + wm_seg

    # back to original SR
    to_orig = Resample(orig_freq=AS_SR, new_freq=orig_sr)
    wm_up = to_orig(wm_buf.unsqueeze(0)).squeeze(0)
    wm_up = ensure_len(wm_up, n_samples)

    # RMS cap
    host = torch.from_numpy(audio_np.mean(axis=1)).float()
    host_r = rms(host)
    wm_r = rms(wm_up)
    if wm_r > rms_cap * max(host_r, 1e-6):
        scale = (rms_cap * max(host_r, 1e-6)) / wm_r
        wm_up = wm_up * float(scale)

    wm_multi = wm_up.unsqueeze(1).repeat(1, n_ch) * float(gain)
    mixed = torch.from_numpy(audio_np) + wm_multi
    # peak normalize
    peak = mixed.abs().max().item()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.cpu().numpy().astype(np.float32, copy=False)

def embed_payload_wavmark(audio_np: np.ndarray, orig_sr: int, payload_bytes: bytes,
                          gain: float, rms_cap: float) -> np.ndarray:
    # Placeholder for integrating a higher‑capacity engine if installed
    try:
        import wavmark  # type: ignore
    except Exception:
        raise SystemExit("Engine 'wavmark' requested but library not installed. Use --engine audioseal (fallback).")
    raise SystemExit("WavMark integration stub: implement encode/decode per your wavmark API/version.")

# --------- Capacity checks ---------

def capacity_bits_audioseal(duration_s: float, seg_sec: float, repeat: int, sync_interval: int) -> int:
    seg_sec = max(MIN_SEG_SEC, seg_sec)
    total_segments = max(1, int(duration_s // seg_sec))
    # frames we can carry ≈ total_segments / repeat minus sync overhead
    frames_capacity = total_segments // max(1, repeat)
    sync_overhead = (frames_capacity // max(1, sync_interval)) + 1 if sync_interval > 0 else 0
    usable_frames = max(0, frames_capacity - sync_overhead)
    return usable_frames * 16

def capacity_bits_wavmark(duration_s: float) -> int:
    # heuristic: ~32 bps robust net (adjust to your model after calibration)
    return int(32 * duration_s)

# --------- Main ---------

def parse_args():
    p = argparse.ArgumentParser(description="Single‑path audio watermark encoder with graceful fallback (hardened).")
    p.add_argument("--in", dest="infile", required=True, help="Input WAV/AIFF/FLAC path.")
    p.add_argument("--out", dest="outfile", default=None, help="Output WAV (default: <in>.wm.wav).")
    p.add_argument("--manifest", help="Path to full JSON manifest. If omitted, auto‑built from flags.")
    p.add_argument("--published-at", default=None, help="ISO datetime for published_at (manifest + micro anchor).")
    p.add_argument("--model", default=None, help="Generator model (manifest + micro anchor).")
    p.add_argument("--issuer", default=None, help="Issuer/organization (manifest + micro anchor).")
    p.add_argument("--title", default=None, help="Optional title (manifest only).")
    p.add_argument("--license", dest="license_str", default=None, help="License (manifest only).")
    p.add_argument("--text-file", default=None, help="Optional transcript file; SHA256 stored in manifest.")
    p.add_argument("--seg", type=float, default=0.25, help="Segment length (s) for streaming (default 0.25; min 0.20).")
    p.add_argument("--repeat", type=int, default=3, help="Repetitions per frame (default 3).")
    p.add_argument("--gain", type=float, default=1.0, help="Watermark gain multiplier.")
    p.add_argument("--rms-cap", type=float, default=0.06, help="Cap WM RMS to fraction of host RMS (default 0.06).")
    p.add_argument("--key-id", type=int, default=1, help="Key identifier (0..255) for schedule seeding.")
    p.add_argument("--key", default="", help="Secret/passphrase used to seed placement schedule.")
    p.add_argument("--auth-key", default="", help="Optional secret to HMAC‑authenticate payload (recommended).")
    p.add_argument("--engine", choices=[ENGINE_WAVMARK, ENGINE_AUDIOSEAL], default=ENGINE_WAVMARK,
                   help="Preferred engine (default wavmark; falls back to audioseal automatically).")
    p.add_argument("--pcm16", action="store_true", help="Write PCM_16 WAV instead of float.")
    p.add_argument("--sync-interval", type=int, default=32, help="Insert one sync frame every N data frames (default 32).")
    p.add_argument("--sync-word", type=lambda x: int(x, 0), default=DEFAULT_SYNC_WORD, help="16‑bit sync word (e.g., 0xB39F).")
    p.add_argument("--energy-gate-db", type=float, default=-20.0, help="Keep segments within this dB of median RMS (default -20dB).")
    p.add_argument("--jitter", type=float, default=0.02, help="Max start jitter (seconds) within a segment (default 0.02s).")
    p.add_argument("--log", default="wm_singlepath_log.csv", help="CSV audit log (append).")
    return p.parse_args()

def main():
    args = parse_args()

    infile = args.infile
    outfile = args.outfile or (os.path.splitext(infile)[0] + ".wm.wav")
    if os.path.abspath(infile) == os.path.abspath(outfile):
        raise SystemExit("Refusing to overwrite input. Use a different --out.")

    audio_np, orig_sr = read_audio(infile)
    n_samples, n_ch = audio_np.shape
    dur = n_samples / float(orig_sr)

    # Get or build full manifest
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)
    else:
        manifest_json = auto_manifest(infile, args.published_at, args.model,
                                      args.title, args.issuer, args.license_str, args.text_file)

    # Seeds & keys
    key_seed = hashlib.sha256((args.key or "").encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
    auth_key = (args.auth_key or "").encode("utf-8")

    # Build payloads
    full_payload = pack_full_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, seed=key_seed)
    full_bits = len(full_payload) * 8

    micro_payload = pack_micro_payload(
        full_manifest_json=manifest_json,
        published_at=args.published_at,
        model=args.model,
        issuer=args.issuer,
        key_id=args.key_id,
        auth_key=auth_key,
        seed=key_seed
    )
    micro_bits = len(micro_payload) * 8

    anchor_payload = pack_anchor_payload(
        full_manifest_json=manifest_json,
        key_id=args.key_id,
        auth_key=auth_key,
        seed=key_seed
    )
    anchor_bits = len(anchor_payload) * 8

    # Engine availability
    preferred_engine = args.engine
    engine_used = preferred_engine
    wavmark_ok = False
    if preferred_engine == ENGINE_WAVMARK:
        try:
            import wavmark  # type: ignore
            wavmark_ok = True
        except Exception:
            engine_used = ENGINE_AUDIOSEAL

    # Capacity estimate
    if engine_used == ENGINE_WAVMARK and wavmark_ok:
        cap_bits = capacity_bits_wavmark(dur)
    else:
        cap_bits = capacity_bits_audioseal(dur, args.seg, args.repeat, args.sync_interval)

    # Decide payload with graceful degradation
    if full_bits <= cap_bits:
        payload_used = "full"
        payload_bytes = full_payload
    elif micro_bits <= cap_bits:
        payload_used = "micro"
        payload_bytes = micro_payload
    elif anchor_bits <= cap_bits:
        payload_used = "anchor"
        payload_bytes = anchor_payload
    else:
        raise SystemExit(f"Not enough capacity even for anchor: duration={dur:.2f}s cap≈{cap_bits} bits, "
                         f"full={full_bits}, micro={micro_bits}, anchor={anchor_bits}. Increase duration or lower repetition.")

    # Embed
    if engine_used == ENGINE_WAVMARK and wavmark_ok:
        out_np = embed_payload_wavmark(audio_np, orig_sr, payload_bytes, args.gain, args.rms_cap)
    else:
        bits = bytes_to_bits(payload_bytes)
        frames = bits_to_frames(bits, width=16)
        out_np = embed_frames_audioseal(
            audio_np, orig_sr, frames, args.seg, args.repeat, args.gain,
            args.rms_cap, key_seed, args.sync_interval, args.sync_word,
            args.energy_gate_db, args.jitter
        )

    # Save
    write_audio(outfile, out_np, orig_sr, pcm16=args.pcm16)

    # Log
    need_header = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","engine","payload_used","in_sr","channels","duration_s",
            "cap_bits","payload_bits","seg","repeat","key_id","sync_interval",
            "energy_gate_db","jitter_s","infile","outfile"
        ])
        if need_header:
            w.writeheader()
        w.writerow({
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": engine_used,
            "payload_used": payload_used,
            "in_sr": orig_sr,
            "channels": n_ch,
            "duration_s": round(dur,3),
            "cap_bits": cap_bits,
            "payload_bits": len(payload_bytes)*8,
            "seg": round(args.seg,3),
            "repeat": args.repeat,
            "key_id": args.key_id,
            "sync_interval": args.sync_interval,
            "energy_gate_db": args.energy_gate_db,
            "jitter_s": args.jitter,
            "infile": os.path.abspath(infile),
            "outfile": os.path.abspath(outfile),
        })

    print(f"Wrote: {outfile} | engine={engine_used} | payload={payload_used} | "
          f"cap≈{cap_bits} bits | used={len(payload_bytes)*8} bits")

if __name__ == "__main__":
    main()
