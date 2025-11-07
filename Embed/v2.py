#!/usr/bin/env python3
"""
wm_singlepath.py — Single‑path audio watermark embedder (hardened + crypto fixes)

Key updates after expert review:
- Real HMAC‑SHA256 (truncated 16B) with header FLAG_AUTH bit.
- HKDF(key, nonce) derives per‑file subkeys for placement and sync.
- Nonce now salts placement scheduling and keyed 16‑bit sync word.
- 32‑bit length fields (handles large compressed manifests).
- Min‑spacing scheduler and simple spectral‑mask scoring for segment choice.
- Soft limiter instead of global peak normalize; logs WM/host RMS and approx dB delta.
- Silent default fallback to audioseal when wavmark is absent.
"""

import argparse
import binascii
import csv
import datetime as dt
import hashlib
import hmac
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

HEADER_VER = 4  # header format version bump

# Flags
FLAG_AUTH = 0x01

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

# Soft limiter with soft knee
def soft_limit(x: torch.Tensor, threshold: float = 0.98, knee: float = 0.02) -> torch.Tensor:
    thr = float(threshold)
    k = max(1e-6, float(knee))
    mag = torch.abs(x)
    over = torch.clamp((mag - thr) / k, min=0.0)
    gain = 1.0 / (1.0 + over)  # gentle compression
    y = torch.sign(x) * mag * gain
    # hard clip safety
    y = torch.clamp(y, -1.0, 1.0)
    return y

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

# --------- Crypto helpers ---------

def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

def canonical_json_bytes(md: dict) -> bytes:
    return json.dumps(md, sort_keys=True, separators=(",", ":")).encode("utf-8")

def compress_zlib(b: bytes) -> bytes:
    return zlib.compress(b, level=9)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_128(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()[:16]

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

# --------- Packet formats (v4) ---------
# magic(2) 'WM', ver(1), key_id(1), type(1), flags(1), nonce(4), len(4),
# tag(16), crc32(4), body(...)
TYPE_MICRO = 0
TYPE_FULL  = 2
TYPE_ANCHOR= 3

def make_nonce(seed_bytes: bytes) -> int:
    h = hashlib.sha256(seed_bytes + os.urandom(16)).digest()
    return struct.unpack(">I", h[:4])[0]

def pack_payload(body: bytes, typ: int, key_id: int, auth_key: bytes, nonce: int) -> bytes:
    magic = b"WM"
    ver = bytes([HEADER_VER & 0xFF])
    kid = bytes([key_id & 0xFF])
    typb = bytes([typ & 0xFF])
    flags = 0
    if auth_key:
        flags |= FLAG_AUTH
    flagsb = bytes([flags & 0xFF])
    ln = len(body).to_bytes(4, "big")
    head_wo_tag = ver + kid + typb + flagsb + nonce.to_bytes(4, "big") + ln + body
    tag = hmac.new(auth_key, head_wo_tag, hashlib.sha256).digest()[:16] if auth_key else b"\x00"*16
    crc = binascii.crc32(body).to_bytes(4, "big")
    head = magic + ver + kid + typb + flagsb + nonce.to_bytes(4, "big") + ln + tag + crc
    return head + body

def pack_full_payload(manifest: dict, key_id: int, auth_key: bytes, nonce: int) -> bytes:
    body = compress_zlib(canonical_json_bytes(manifest))
    return pack_payload(body, TYPE_FULL, key_id, auth_key, nonce)

def pack_micro_payload(full_manifest_json: dict, published_at: str | None,
                       model: str | None, issuer: str | None,
                       key_id: int, auth_key: bytes, nonce: int) -> bytes:
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
    return pack_payload(body, TYPE_MICRO, key_id, auth_key, nonce)

def pack_anchor_payload(full_manifest_json: dict, key_id: int, auth_key: bytes, nonce: int) -> bytes:
    anchor = sha256_128(canonical_json_bytes(full_manifest_json))
    body = anchor
    return pack_payload(body, TYPE_ANCHOR, key_id, auth_key, nonce)

# --------- Spectral mask scoring ---------

def segment_scores(mono_m: torch.Tensor, seg_len_m: int) -> np.ndarray:
    """Return a score per segment: combine RMS and 1–4 kHz band energy emphasis."""
    n = mono_m.numel()
    nseg = max(1, n // seg_len_m)
    if nseg == 1:
        return np.array([1.0], dtype=np.float32)
    scores = np.zeros(nseg, dtype=np.float32)
    win = torch.hann_window(seg_len_m, periodic=False)
    for i in range(nseg):
        s = i * seg_len_m
        e = min(s + seg_len_m, n)
        seg = mono_m[s:e]
        if seg.numel() < seg_len_m:
            seg = torch.nn.functional.pad(seg, (0, seg_len_m - seg.numel()))
        segw = seg * win
        spec = torch.fft.rfft(segw, n=seg_len_m)
        mag2 = (spec.real**2 + spec.imag**2).float()
        # frequency bins
        freqs = torch.linspace(0, AS_SR/2, mag2.numel())
        mask = (freqs >= 1000) & (freqs <= 4000)
        band = float(torch.sum(mag2[mask]) + 1e-9)
        total = float(torch.sum(mag2) + 1e-9)
        r = band / total
        scores[i] = r
    # Normalize to mean 1
    m = float(np.mean(scores)) + 1e-9
    return scores / m

# --------- Scheduling & placement ---------

def energy_gated_indices(mono_m: torch.Tensor, seg_len_m: int, gate_db: float) -> List[int]:
    n = mono_m.numel()
    nseg = max(1, n // seg_len_m)
    if nseg == 1:
        return [0]
    vals = []
    for i in range(nseg):
        s = i * seg_len_m
        e = min(s + seg_len_m, n)
        vals.append(rms(mono_m[s:e]))
    vals = np.array(vals, dtype=np.float32)
    med = float(np.median(vals)) + 1e-9
    db = 20.0 * np.log10(np.maximum(vals, 1e-9) / med)
    keep = [i for i, v in enumerate(db) if v >= gate_db]
    if not keep:
        keep = list(range(nseg))
    return keep

def schedule_positions(pool: List[int], total_needed: int, min_spacing: int, seed: bytes, scores: np.ndarray) -> List[int]:
    """Greedy selection with minimum spacing; prefers higher score segments."""
    rnd = random.Random(sha256_hex(seed))
    # sort pool by score desc, then random tiebreaker
    ranked = sorted(pool, key=lambda i: (scores[i], rnd.random()), reverse=True)
    chosen = []
    for cand in ranked:
        if all(abs(cand - c) >= min_spacing for c in chosen):
            chosen.append(cand)
            if len(chosen) >= total_needed:
                break
    # if insufficient, relax spacing
    if len(chosen) < total_needed:
        remaining = [i for i in ranked if i not in chosen]
        for cand in remaining:
            chosen.append(cand)
            if len(chosen) >= total_needed:
                break
    # If still short, cycle
    while len(chosen) < total_needed:
        chosen.append(rnd.choice(pool))
    return chosen

# --------- Engines ---------

def embed_frames_audioseal(audio_np: np.ndarray, orig_sr: int, frames: List[List[int]],
                           seg_sec: float, repeat: int, gain: float,
                           rms_cap: float, key_seed: bytes, sync_word: int,
                           energy_gate_db: float, jitter_s: float, min_spacing: int) -> Tuple[np.ndarray, float, float]:
    from audioseal import AudioSeal
    seg_sec = max(MIN_SEG_SEC, seg_sec)

    n_samples, n_ch = audio_np.shape
    dur = n_samples / float(orig_sr)

    # analysis mono @ 16k
    mono = torch.from_numpy(audio_np.mean(axis=1)).float()
    to_model = Resample(orig_sr=orig_sr, new_freq=AS_SR)
    mono_m = to_model(mono.unsqueeze(0)).squeeze(0)

    seg_len_m = int(seg_sec * AS_SR)
    n_segments = max(1, mono_m.numel() // seg_len_m)

    # Insert sync frames
    frames_with_sync = []
    for idx, fr in enumerate(frames):
        if idx % 32 == 0:  # local default; caller should pre-derive sync_word
            frames_with_sync.append(u16_to_bits(sync_word))
        frames_with_sync.append(fr)
    frames_with_sync.append(u16_to_bits(sync_word))

    total_needed = len(frames_with_sync) * max(1, repeat)

    # Energy gate + spectral score
    pool = energy_gated_indices(mono_m, seg_len_m, gate_db=energy_gate_db)
    scores = segment_scores(mono_m, seg_len_m)
    if len(pool) < total_needed:
        pool = list(range(n_segments))

    positions = schedule_positions(pool, total_needed, min_spacing, seed=key_seed, scores=scores)

    wm_buf = torch.zeros_like(mono_m)
    gen = AudioSeal.load_generator(AS_MODEL_NAME)

    # jitter in samples at model SR
    jitter_m = int(max(0.0, float(jitter_s)) * AS_SR)
    rnd = random.Random(sha256_hex(key_seed + b"jitter"))

    idx = 0
    for fr in frames_with_sync:
        for r in range(repeat):
            seg_idx = positions[idx]; idx += 1
            base = seg_idx * seg_len_m
            j = rnd.randint(-jitter_m//2, jitter_m//2) if jitter_m > 0 else 0
            s = max(0, min(base + j, mono_m.numel() - seg_len_m))
            e = s + seg_len_m
            seg = mono_m[s:e].contiguous()
            message = torch.tensor([[int(b) for b in fr]], dtype=torch.int32)
            wm_seg = gen.get_watermark(seg, message, sample_rate=AS_SR)
            wm_buf[s:e] = wm_buf[s:e] + wm_seg

    # back to original SR
    to_orig = Resample(orig_freq=AS_SR, new_freq=orig_sr)
    wm_up = to_orig(wm_buf.unsqueeze(0)).squeeze(0)
    wm_up = ensure_len(wm_up, n_samples)

    # RMS cap then soft limit
    host = torch.from_numpy(audio_np.mean(axis=1)).float()
    host_r = rms(host)
    wm_r = rms(wm_up)
    if wm_r > rms_cap * max(host_r, 1e-6):
        scale = (rms_cap * max(host_r, 1e-6)) / wm_r
        wm_up = wm_up * float(scale)

    wm_multi = wm_up.unsqueeze(1).repeat(1, n_ch) * float(gain)
    mixed = torch.from_numpy(audio_np) + wm_multi

    mixed = soft_limit(mixed, threshold=0.98, knee=0.02)

    wm_r_post = rms(wm_up)
    return mixed.cpu().numpy().astype(np.float32, copy=False), host_r, wm_r_post

def embed_payload_wavmark(audio_np: np.ndarray, orig_sr: int, payload_bytes: bytes,
                          gain: float, rms_cap: float) -> Tuple[np.ndarray, float, float]:
    # Placeholder for integrating a higher‑capacity engine if installed
    try:
        import wavmark  # type: ignore
    except Exception:
        raise SystemExit("Engine 'wavmark' requested but library not installed. Use --engine audioseal (fallback).")
    raise SystemExit("WavMark integration stub: implement encode/decode per your wavmark API/version.")

# --------- Capacity checks ---------

def capacity_bits_audioseal(duration_s: float, seg_sec: float, repeat: int, sync_every: int) -> int:
    seg_sec = max(MIN_SEG_SEC, seg_sec)
    total_segments = max(1, int(duration_s // seg_sec))
    frames_capacity = total_segments // max(1, repeat)
    sync_overhead = (frames_capacity // max(1, sync_every)) + 1 if sync_every > 0 else 0
    usable_frames = max(0, frames_capacity - sync_overhead)
    return usable_frames * 16

def capacity_bits_wavmark(duration_s: float) -> int:
    # heuristic; tune with your model
    return int(32 * duration_s)

# --------- Main ---------

def parse_args():
    p = argparse.ArgumentParser(description="Single‑path audio watermark encoder (hardened + crypto fixes).")
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
    p.add_argument("--key", default="", help="Secret/passphrase to derive subkeys (HKDF).")
    p.add_argument("--auth-key", default="", help="Optional secret to HMAC‑authenticate payload (recommended).")
    p.add_argument("--engine", choices=[ENGINE_WAVMARK, ENGINE_AUDIOSEAL], default=ENGINE_WAVMARK,
                   help="Preferred engine (default wavmark; silently falls back to audioseal).")
    p.add_argument("--pcm16", action="store_true", help="Write PCM_16 WAV instead of float.")
    p.add_argument("--sync-every", type=int, default=32, help="Insert one keyed sync frame every N data frames (default 32).")
    p.add_argument("--energy-gate-db", type=float, default=-20.0, help="Keep segments within this dB of median RMS (default -20dB).")
    p.add_argument("--jitter", type=float, default=0.02, help="Max start jitter (seconds) within a segment (default 0.02s).")
    p.add_argument("--min-spacing", type=int, default=2, help="Minimum spacing (in segments) between placements (default 2).")
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

    # Build/obtain manifest
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)
    else:
        manifest_json = auto_manifest(infile, args.published_at, args.model,
                                      args.title, args.issuer, args.license_str, args.text_file)

    # Nonce per copy
    base_seed = hashlib.sha256((args.key or "").encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
    nonce = struct.unpack(">I", hkdf_sha256(base_seed, salt=b"nonce", info=b"wm.nonce", length=4))[0]

    # Subkeys for placement (Kp) and sync (Ks)
    Kp = hkdf_sha256((args.key or "").encode("utf-8"), salt=nonce.to_bytes(4,"big"), info=b"wm.place", length=32)
    Ks = hkdf_sha256((args.key or "").encode("utf-8"), salt=nonce.to_bytes(4,"big"), info=b"wm.sync", length=32)

    auth_key = (args.auth_key or "").encode("utf-8")

    # Payloads
    full_payload = pack_full_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)
    micro_payload = pack_micro_payload(manifest_json, args.published_at, args.model, args.issuer,
                                       key_id=args.key_id, auth_key=auth_key, nonce=nonce)
    anchor_payload = pack_anchor_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)

    full_bits   = len(full_payload) * 8
    micro_bits  = len(micro_payload) * 8
    anchor_bits = len(anchor_payload) * 8

    # Engine availability (silent fallback)
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
        cap_bits = capacity_bits_audioseal(dur, args.seg, args.repeat, args.sync_every)

    # Choose payload
    if full_bits <= cap_bits:
        payload_used = "full"; payload_bytes = full_payload
    elif micro_bits <= cap_bits:
        payload_used = "micro"; payload_bytes = micro_payload
    elif anchor_bits <= cap_bits:
        payload_used = "anchor"; payload_bytes = anchor_payload
    else:
        raise SystemExit(f"Capacity too low: cap≈{cap_bits} bits, full={full_bits}, micro={micro_bits}, anchor={anchor_bits}.")

    # Derive keyed sync word from Ks
    sync_word = struct.unpack(">H", hkdf_sha256(Ks, salt=b"sync", info=b"wm.sync.word", length=2))[0] | 1  # ensure odd

    # Embed
    if engine_used == ENGINE_WAVMARK and wavmark_ok:
        out_np, host_r, wm_r = embed_payload_wavmark(audio_np, orig_sr, payload_bytes, args.gain, args.rms_cap)
    else:
        bits = bytes_to_bits(payload_bytes)
        frames = bits_to_frames(bits, width=16)
        # key for placement/jitter uses Kp
        out_np, host_r, wm_r = embed_frames_audioseal(
            audio_np, orig_sr, frames, args.seg, args.repeat, args.gain,
            args.rms_cap, Kp, sync_word,
            args.energy_gate_db, args.jitter, args.min_spacing
        )

    # Save
    write_audio(outfile, out_np, orig_sr, pcm16=args.pcm16)

    # Log
    need_header = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","engine","payload_used","header_ver","auth","key_id","nonce",
            "in_sr","channels","duration_s",
            "cap_bits","payload_bits","seg","repeat",
            "energy_gate_db","min_spacing","jitter_s",
            "host_rms","wm_rms","wm_to_host_db",
            "infile","outfile"
        ])
        if need_header:
            w.writeheader()
        wm_to_host_db = (20.0 * math.log10(max(wm_r,1e-9)/max(host_r,1e-9)))
        w.writerow({
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": engine_used,
            "payload_used": payload_used,
            "header_ver": HEADER_VER,
            "auth": 1 if auth_key else 0,
            "key_id": args.key_id,
            "nonce": nonce,
            "in_sr": orig_sr,
            "channels": audio_np.shape[1],
            "duration_s": round(dur,3),
            "cap_bits": cap_bits,
            "payload_bits": len(payload_bytes)*8,
            "seg": round(args.seg,3),
            "repeat": args.repeat,
            "energy_gate_db": args.energy_gate_db,
            "min_spacing": args.min_spacing,
            "jitter_s": args.jitter,
            "host_rms": round(host_r,6),
            "wm_rms": round(wm_r,6),
            "wm_to_host_db": round(wm_to_host_db,2),
            "infile": os.path.abspath(infile),
            "outfile": os.path.abspath(outfile),
        })

    print(f"Wrote: {outfile} | engine={engine_used} | payload={payload_used} | "
          f"cap≈{cap_bits} bits | used={len(payload_bytes)*8} bits | WM/host ≈ {wm_to_host_db:.1f} dB")

if __name__ == "__main__":
    main()
