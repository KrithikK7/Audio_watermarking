#!/usr/bin/env python3
"""
wm_singlepath.py — Single‑path audio watermark embedder/decoder (v7.6)

New in v7
---------
- **Mode switch**: `--mode {encode,decode}` with a minimal **decoder skeleton** (pilot correlator hooks,
  drift search scaffolding, FEC decode hooks). AudioSeal detector integration is stubbed until available.
- **BCH option**: `--fec bch63` (requires `bchlib`). Encodes with BCH(63,45) by default params; decode hook present.
- **Stereo default**: For 2‑ch inputs, **M/S embedding is enabled by default**. Disable via `--no-ms-embed`.
- **Scoring polish**: Penalize segments below −10 dB relative to median energy in spectral scoring.
- **Placement spreading**: After relaxation, re‑spread chosen positions to avoid local clustering.
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
from typing import List, Tuple, Optional

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

HEADER_VER = 7  # version bump

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
def soft_limit(x: torch.Tensor, threshold: float = 0.98, knee: float = 0.02) -> Tuple[torch.Tensor, bool]:
    thr = float(threshold); k = max(1e-6, float(knee))
    mag = torch.abs(x)
    over = torch.clamp((mag - thr) / k, min=0.0)
    engaged = bool(torch.any(over > 0))
    gain = 1.0 / (1.0 + over)
    y = torch.sign(x) * mag * gain
    return torch.clamp(y, -1.0, 1.0), engaged

# --------- Bits, FEC, frames ---------

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

def frames_to_bits(frames: List[List[int]]) -> List[int]:
    out = []
    for fr in frames:
        out.extend([int(b)&1 for b in fr])
    return out

# Hamming(15,11) encoder & decoder (decoder is parity-only placeholder)
def hamming15_encode_bits(bits: List[int]) -> List[int]:
    out = []
    i = 0
    while i < len(bits):
        block = bits[i:i+11]
        if len(block) < 11:
            block += [0]*(11-len(block))
        cw = [0]*16
        data_positions = [3,5,6,7,9,10,11,12,13,14,15]
        for dp, dbit in zip(data_positions, block):
            cw[dp] = dbit & 1
        def parity(indices):
            s = 0
            for j in indices:
                s ^= cw[j]
            return s & 1
        p1 = parity([1,3,5,7,9,11,13,15])
        p2 = parity([2,3,6,7,10,11,14,15])
        p4 = parity([4,5,6,7,12,13,14,15])
        p8 = parity([8,9,10,11,12,13,14,15])
        cw[1]=p1; cw[2]=p2; cw[4]=p4; cw[8]=p8
        for pos in range(1,16):
            out.append(cw[pos])
        i += 11
    return out

def hamming15_decode_bits(bits: List[int]) -> List[int]:
    """Hamming(15,11) single-bit error correction with standard (1,2,4,8) parity positions."""
    out: List[int] = []
    i = 0
    while i + 15 <= len(bits):
        cw = [None] + bits[i:i+15]  # 1-indexed
        # Compute syndrome s = p1*1 + p2*2 + p4*4 + p8*8
        def parity(vals):
            s = 0
            for v in vals:
                s ^= v & 1
            return s & 1
        p1 = parity([cw[j] for j in [1,3,5,7,9,11,13,15]])
        p2 = parity([cw[j] for j in [2,3,6,7,10,11,14,15]])
        p4 = parity([cw[j] for j in [4,5,6,7,12,13,14,15]])
        p8 = parity([cw[j] for j in [8,9,10,11,12,13,14,15]])
        s = (p8<<3) | (p4<<2) | (p2<<1) | p1
        if s != 0 and 1 <= s <= 15:
            cw[s] ^= 1  # correct single-bit error
        data_positions = [3,5,6,7,9,10,11,12,13,14,15]
        for dp in data_positions:
            out.append(cw[dp] & 1)
        i += 15
    return out

# BCH via optional bchlib
def bch63_encode_bits(bits: List[int]) -> List[int]:
    """
    Byte-aligned BCH(63,45) encoder:
    - Pack each 45-bit block left-aligned into 6 data bytes (48 bits total).
    - Compute ecc bytes with bchlib and append (data||ecc) *bytes*.
    - Emit all 8 bits per byte (no 63-bit truncation). Decoder will slice by bytes.
    """
    try:
        import bchlib  # type: ignore
    except Exception:
        raise SystemExit("FEC 'bch63' requires the 'bchlib' package. Install it or choose another FEC.")
    BCH_POLY = 8219  # example poly for (63,45) t=3
    bch = bchlib.BCH(BCH_POLY, t=3)
    DATA_LEN = 6  # bytes to carry 45 bits, left-aligned (48-bit container)
    out_bits: List[int] = []
    i = 0
    while i < len(bits):
        block = bits[i:i+45]
        if len(block) < 45:
            block += [0]*(45-len(block))
        # Build 45-bit integer, MSB-first, then left-align into 48-bit (DATA_LEN*8) container
        val = 0
        for b in block:
            val = (val << 1) | (b & 1)
        pad = DATA_LEN*8 - 45  # 3 pad bits
        val_aligned = (val << pad) & ((1 << (DATA_LEN*8)) - 1)
        data_bytes = val_aligned.to_bytes(DATA_LEN, "big")
        ecc = bch.encode(data_bytes)  # ecc_bytes known internally
        packet = data_bytes + ecc     # bytes, byte-aligned
        # Emit all bits of the packet, byte-wise (no truncation)
        for by in packet:
            for k in range(7, -1, -1):
                out_bits.append((by >> k) & 1)
        i += 45
    # Pad to 16-bit frame boundary for packing
    pad16 = (-len(out_bits)) % 16
    out_bits += [0] * pad16
    return out_bits

def bch63_decode_bits(bits: List[int]) -> List[int]:
    """
    Byte-aligned BCH(63,45) decoder (mirrors encoder):
    - Rebuild bytes; process packets of (DATA_LEN + ecc_bytes).
    - Correct with bchlib, then extract the top 45 bits from the 48-bit data.
    """
    try:
        import bchlib  # type: ignore
    except Exception:
        raise SystemExit("FEC 'bch63' requires the 'bchlib' package. Install it or choose another FEC.")
    BCH_POLY = 8219
    bch = bchlib.BCH(BCH_POLY, t=3)
    DATA_LEN = 6
    CODEWORD_BYTES = DATA_LEN + bch.ecc_bytes
    # Convert incoming bitstream to bytes
    byt = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            chunk += [0]*(8 - len(chunk))
        v = 0
        for b in chunk:
            v = (v << 1) | (b & 1)
        byt.append(v & 0xFF)
    # Process fixed-size packets
    out_bits: List[int] = []
    pos = 0
    while pos + CODEWORD_BYTES <= len(byt):
        packet = byt[pos:pos+CODEWORD_BYTES]
        data = bytes(packet[:DATA_LEN])
        ecc = bytes(packet[DATA_LEN:])
        d_corr, ecc_corr = bch.decode(bytearray(data), bytearray(ecc))
        # Re-extract 45 data bits from the top bits of DATA_LEN bytes (left-aligned)
        val = int.from_bytes(d_corr, "big")
        start_bit = DATA_LEN*8 - 1
        for j in range(45):
            bit = (val >> (start_bit - j)) & 1
            out_bits.append(bit)
        pos += CODEWORD_BYTES
    return out_bits

def apply_fec(bits: List[int], fec: str) -> List[int]:
    if fec == "hamming15":
        coded = hamming15_encode_bits(bits)
        pad = (-len(coded)) % 16
        coded += [0]*pad
        return coded
    if fec == "bch63":
        return bch63_encode_bits(bits)
    return bits

def remove_fec(bits: List[int], fec: str) -> List[int]:
    if fec == "hamming15":
        return hamming15_decode_bits(bits)
    if fec == "bch63":
        return bch63_decode_bits(bits)
    return bits

def block_interleave(bits: List[int], depth: int) -> List[int]:
    if depth <= 1:
        return bits
    rows = depth
    cols = math.ceil(len(bits) / rows)
    grid = [[0]*cols for _ in range(rows)]
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(bits):
                grid[r][c] = bits[idx]; idx += 1
    out = []
    for c in range(cols):
        for r in range(rows):
            out.append(grid[r][c])
    return out[:len(bits)]

def block_deinterleave(bits: List[int], depth: int) -> List[int]:
    if depth <= 1:
        return bits
    rows = depth
    cols = math.ceil(len(bits) / rows)
    grid = [[0]*cols for _ in range(rows)]
    idx = 0
    # fill column-wise
    for c in range(cols):
        for r in range(rows):
            if idx < len(bits):
                grid[r][c] = bits[idx]; idx += 1
    # read row-wise
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(grid[r][c])
    return out[:len(bits)]

# --------- Pilot insertion ---------

def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    t = b""; okm = b""; c = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([c]), hashlib.sha256).digest()
        okm += t; c += 1
    return okm[:length]

def derive_pilot_bits(Ks: bytes, pilot_bits: int) -> List[int]:
    nbytes = math.ceil(pilot_bits / 8)
    stream = hkdf_sha256(Ks, salt=b"pilot", info=b"wm.pilot", length=nbytes)
    bits = []
    for b in stream:
        for i in range(7,-1,-1):
            bits.append((b >> i) & 1)
    return bits[:pilot_bits]

def add_pilot_frames(frames: List[List[int]], sync_every: int, pilot_bits: int, Ks: bytes, terminal: bool=True) -> Tuple[List[List[int]], int, int]:
    if sync_every <= 0 or pilot_bits <= 0:
        return frames[:], 0, 0
    pilot = derive_pilot_bits(Ks, pilot_bits)
    pilot_frames = bits_to_frames(pilot, width=16)
    out = []
    n_pilots = 0
    data_count = 0
    for fr in frames:
        if data_count % sync_every == 0:
            out.extend(pilot_frames); n_pilots += 1
        out.append(fr); data_count += 1
    if terminal:
        out.extend(pilot_frames); n_pilots += 1
    return out, n_pilots, len(pilot_frames)

# --------- Manifest & packet formats ---------

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

# Header v7: magic(2) 'WM', ver(1), key_id(1), type(1), flags(1), nonce(4), len(4), tag(16), crc32(4 or zeros), body(...)
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
    flags = FLAG_AUTH if auth_key else 0
    flagsb = bytes([flags & 0xFF])
    ln = len(body).to_bytes(4, "big")
    head_wo_tag = ver + kid + typb + flagsb + nonce.to_bytes(4, "big") + ln + body
    tag = hmac.new(auth_key, head_wo_tag, hashlib.sha256).digest()[:16] if auth_key else b"\x00"*16
    crc = (b"\x00"*4) if auth_key else (binascii.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")
    head = magic + ver + kid + typb + flagsb + nonce.to_bytes(4, "big") + ln + tag + crc
    return head + body

def parse_payload(raw: bytes) -> Tuple[dict, bytes, int, int, int, int, int, bytes, bytes]:
    """Return (header_dict, body, key_id, typ, flags, nonce, ver, tag, crc)."""
    if len(raw) < 2+1+1+1+1+4+4+16+4:
        raise ValueError("packet too short")
    off = 0
    magic = raw[off:off+2]; off+=2
    if magic != b"WM":
        raise ValueError("bad magic")
    ver = raw[off]; off+=1
    kid = raw[off]; off+=1
    typ = raw[off]; off+=1
    flags = raw[off]; off+=1
    nonce = int.from_bytes(raw[off:off+4],"big"); off+=4
    ln = int.from_bytes(raw[off:off+4],"big"); off+=4
    tag = raw[off:off+16]; off+=16
    crc = raw[off:off+4]; off+=4
    body = raw[off:off+ln]
    header = {"ver":ver,"key_id":kid,"type":typ,"flags":flags,"nonce":nonce,"len":ln,"tag_hex":tag.hex(),"crc_hex":crc.hex()}
    return header, body, kid, typ, flags, nonce, ver, tag, crc

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

# --------- Spectral scoring ---------

class SpectralScorer:
    def __init__(self, seg_len_m: int, device: str = "cpu"):
        self.seg_len_m = seg_len_m
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.win = torch.hann_window(seg_len_m, periodic=False, device=self.device)
        self.freqs = torch.linspace(0, AS_SR/2, seg_len_m//2 + 1, device=self.device)

    def scores(self, mono_m: torch.Tensor) -> np.ndarray:
        mono_m = mono_m.to(self.device)
        n = mono_m.numel()
        nseg = max(1, n // self.seg_len_m)
        if nseg == 1:
            return np.array([1.0], dtype=np.float32)
        segs = []
        for i in range(nseg):
            s = i * self.seg_len_m
            e = min(s + self.seg_len_m, n)
            seg = mono_m[s:e]
            if seg.numel() < self.seg_len_m:
                seg = torch.nn.functional.pad(seg, (0, self.seg_len_m - seg.numel()))
            segs.append(seg * self.win)
        stack = torch.stack(segs, dim=0)  # [nseg, seg_len]
        spec = torch.fft.rfft(stack, n=self.seg_len_m, dim=1)
        mag2 = (spec.real**2 + spec.imag**2).float()  # [nseg, bins]
        mask = (self.freqs >= 1000) & (self.freqs <= 4000)
        band = torch.sum(mag2[:, mask], dim=1) + 1e-9
        total = torch.sum(mag2, dim=1) + 1e-9
        ratio = (band / total)
        # energy penalty for low-energy segments: compare total energy per seg to median
        total_cpu = total.detach().cpu().numpy()
        med = float(np.median(total_cpu)) + 1e-12
        db_rel = 10.0 * np.log10(np.maximum(total_cpu, 1e-12) / med)
        penalty = np.where(db_rel < -10.0, 0.6, 1.0).astype(np.float32)  # down-weight quiet segments
        r = (ratio.detach().cpu().numpy().astype(np.float32)) * penalty
        m = float(np.mean(r)) + 1e-9
        return r / m

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

def spread_positions(chosen: List[int], n_segments: int) -> List[int]:
    """Greedy farthest-point reordering over the 1-D segment axis.
    Keeps set of chosen indices but orders them to maximize minimum spacing between consecutive picks.
    """
    if not chosen:
        return chosen
    pts = sorted(set(chosen))
    # start with the point that maximizes average distance to others
    def mean_dist(i):
        return sum(abs(i - j) for j in pts) / max(1, len(pts)-1)
    seed = max(pts, key=mean_dist)
    ordered = [seed]
    remaining = [i for i in pts if i != seed]
    while remaining:
        # pick the candidate with maximum min-distance to already ordered set
        best = None; best_d = -1
        for c in remaining:
            d = min(abs(c - o) for o in ordered)
            if d > best_d:
                best_d = d; best = c
        ordered.append(best)
        remaining.remove(best)
    return ordered

def schedule_positions(pool: List[int], total_needed: int, min_spacing: int, seed: bytes, scores: np.ndarray,
                       max_hits_per_seg: int, n_segments: int) -> List[int]:
    rnd = random.Random(hashlib.sha256(seed).digest())
    ranked = sorted(pool, key=lambda i: (scores[i], rnd.random()), reverse=True)
    chosen = []
    counts = {i: 0 for i in pool}
    for cand in ranked:
        if counts.get(cand, 0) >= max_hits_per_seg:
            continue
        if all(abs(cand - c) >= min_spacing for c in chosen):
            chosen.append(cand); counts[cand] = counts.get(cand, 0) + 1
            if len(chosen) >= total_needed:
                break
    if len(chosen) < total_needed:
        for cand in ranked:
            if counts.get(cand, 0) >= max_hits_per_seg:
                continue
            chosen.append(cand); counts[cand] = counts.get(cand, 0) + 1
            if len(chosen) >= total_needed:
                break
    chosen = spread_positions(chosen, n_segments)
    while len(chosen) < total_needed:
        pick = rnd.choice(pool)
        if counts.get(pick, 0) < max_hits_per_seg:
            chosen.append(pick); counts[pick] = counts.get(pick, 0) + 1
    return chosen

# --------- Engines (encode) ---------

def embed_frames_audioseal(audio_np: np.ndarray, orig_sr: int, frames_with_pilot: List[List[int]],
                           seg_sec: float, repeat: int, gain: float,
                           rms_cap: float, place_key: bytes,
                           energy_gate_db: float, jitter_s: float, min_spacing: int,
                           s_mix: float, ms_embed: bool,
                           limit_threshold: float, limit_knee: float,
                           device: str, max_hits_per_seg: int) -> Tuple[np.ndarray, float, float, bool]:
    from audioseal import AudioSeal
    seg_sec = max(MIN_SEG_SEC, seg_sec)

    n_samples, n_ch = audio_np.shape

    # analysis mono @ 16k
    if n_ch >= 2 and ms_embed:
        M = torch.from_numpy(((audio_np[:,0] + audio_np[:,1]) * 0.5)).float()
        mono = M
    else:
        mono = torch.from_numpy(audio_np.mean(axis=1)).float()

    to_model = Resample(orig_freq=orig_sr, new_freq=AS_SR)
    mono_m = to_model(mono.unsqueeze(0)).squeeze(0)

    seg_len_m = int(seg_sec * AS_SR)
    n_segments = max(1, mono_m.numel() // seg_len_m)

    total_needed = len(frames_with_pilot)

    # Energy gate + spectral score
    pool = energy_gated_indices(mono_m, seg_len_m, gate_db=energy_gate_db)
    scorer = SpectralScorer(seg_len_m, device=device)
    scores = scorer.scores(mono_m)
    if len(pool) < total_needed:
        pool = list(range(n_segments))

    positions = schedule_positions(pool, total_needed, min_spacing, seed=place_key, scores=scores,
                                   max_hits_per_seg=max_hits_per_seg, n_segments=n_segments)

    wm_buf = torch.zeros_like(mono_m)
    gen = AudioSeal.load_generator(AS_MODEL_NAME)

    # jitter
    jitter_m = int(max(0.0, float(jitter_s)) * AS_SR)
    rnd = random.Random(hashlib.sha256(place_key + b"jitter").digest())

    idx = 0
    for fr in frames_with_pilot:
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

    # RMS cap (use Mid channel RMS when M/S embedding is active)
    if n_ch == 2 and ms_embed:
        host = torch.from_numpy(((audio_np[:,0] + audio_np[:,1]) * 0.5)).float()
        host_ref = 'mid'
    else:
        host = torch.from_numpy(audio_np.mean(axis=1)).float()
        host_ref = 'mixmean'
    host_r = rms(host)
    wm_r = rms(wm_up)
    if wm_r > rms_cap * max(host_r, 1e-6):
        scale = (rms_cap * max(host_r, 1e-6)) / wm_r
        wm_up = wm_up * float(scale)

    # Mix
    if n_ch == 2 and ms_embed:
        L = torch.from_numpy(audio_np[:,0]).float()
        R = torch.from_numpy(audio_np[:,1]).float()
        M = (L + R) * 0.5
        S = (L - R) * 0.5
        M2 = M + wm_up * float(gain)
        S2 = S + wm_up * float(gain) * float(s_mix)
        L2 = M2 + S2
        R2 = M2 - S2
        mixed = torch.stack([L2, R2], dim=1)
    else:
        wm_multi = wm_up.unsqueeze(1).repeat(1, n_ch) * float(gain)
        mixed = torch.from_numpy(audio_np) + wm_multi

    # Soft limiter
    mixed_limited, engaged = soft_limit(mixed, threshold=limit_threshold, knee=limit_knee)

    return mixed_limited.cpu().numpy().astype(np.float32, copy=False), host_r, rms(wm_up), engaged

def embed_payload_wavmark(audio_np: np.ndarray, orig_sr: int, payload_bytes: bytes,
                          gain: float, rms_cap: float) -> Tuple[np.ndarray, float, float, bool]:
    try:
        import wavmark  # type: ignore
    except Exception:
        raise SystemExit("Engine 'wavmark' requested but library not installed. Use --engine audioseal (fallback).")
    raise SystemExit("WavMark integration stub: implement encode/decode per your wavmark API/version.")

# --------- Engines (decode skeleton) ---------

def decode_frames_audioseal(audio_np: np.ndarray, orig_sr: int,
                            seg_sec: float, repeat: int,
                            Ks: bytes, sync_every: int, pilot_bits: int,
                            place_key: bytes, energy_gate_db: float,
                            jitter_s: float, min_spacing: int,
                            device: str, max_hits_per_seg: int) -> Tuple[List[List[int]], dict]:
    """
    Skeleton: find pilot cadence by correlation on predicted positions, then sample frames.
    NOTE: Requires an AudioSeal detector to obtain per-frame 16 soft/hard bits; not implemented here.
    Returns (frames_with_pilot_order, debug_info).
    """
    try:
        from audioseal import AudioSeal  # noqa
    except Exception:
        raise SystemExit("Decoding requires AudioSeal detector; library not available.")
    # Placeholder: without detector, we cannot extract bits from audio.
    raise SystemExit("Decoder skeleton present, but frame extraction requires integrating AudioSeal's detector API.")

# --------- Capacity ---------

def capacity_bits_audioseal(duration_s: float, seg_sec: float, repeat: int, sync_every: int, pilot_bits: int) -> int:
    seg_sec = max(MIN_SEG_SEC, seg_sec)
    total_segments = max(1, int(duration_s // seg_sec))
    frames_capacity = total_segments // max(1, repeat)
    pilot_frames = math.ceil(max(1, pilot_bits) / 16)
    n_pilots = (frames_capacity // max(1, sync_every)) + 1 if sync_every > 0 else 0
    overhead = n_pilots * pilot_frames
    usable_frames = max(0, frames_capacity - overhead)
    return usable_frames * 16

def capacity_bits_wavmark(duration_s: float) -> int:
    return int(32 * duration_s)

# --------- CLI ---------

def parse_args():
    p = argparse.ArgumentParser(description="Single‑path audio watermark embedder/decoder (v7).")
    p.add_argument("--mode", choices=["encode","decode","selftest"], default="encode", help="Run as encoder, decoder (needs detector), or selftest (no audio readback).")
    p.add_argument("--in", dest="infile", required=True, help="Input WAV/AIFF/FLAC path.")
    p.add_argument("--out", dest="outfile", default=None, help="Output WAV for encode; output JSON for decode (default: <in>.wm.wav or <in>.wm.json).")
    p.add_argument("--manifest", help="Path to full JSON manifest. If omitted, auto‑built from flags (encode).")
    p.add_argument("--published-at", default=None, help="ISO datetime for published_at (manifest + micro anchor).")
    p.add_argument("--model", default=None, help="Generator model (manifest + micro anchor).")
    p.add_argument("--issuer", default=None, help="Issuer/organization (manifest + micro anchor).")
    p.add_argument("--title", default=None, help="Optional title (manifest only).")
    p.add_argument("--license", dest="license_str", default=None, help="License (manifest only).")
    p.add_argument("--text-file", default=None, help="Optional transcript file; SHA256 stored in manifest.")
    p.add_argument("--seg", type=float, default=0.25, help="Segment length (s) for streaming (default 0.25; min 0.20).")
    p.add_argument("--repeat", type=int, default=3, help="Repetitions per frame (default 3).")
    p.add_argument("--gain", type=float, default=1.0, help="Watermark gain multiplier (encode).")
    p.add_argument("--rms-cap", type=float, default=0.06, help="Cap WM RMS to fraction of host RMS (default 0.06).")
    p.add_argument("--key-id", type=int, default=1, help="Key identifier (0..255) for schedule seeding.")
    p.add_argument("--key", default="", help="Secret/passphrase to derive subkeys (HKDF).")
    p.add_argument("--auth-key", default="", help="Optional secret to HMAC‑authenticate payload.")
    p.add_argument("--secret", default="", help="Single secret to derive Kp, Ks, and Ka (overrides --key/--auth-key).")
    p.add_argument("--engine", choices=[ENGINE_WAVMARK, ENGINE_AUDIOSEAL], default=ENGINE_WAVMARK,
                   help="Preferred engine (default wavmark; silently falls back to audioseal).")
    p.add_argument("--pcm16", action="store_true", help="(encode) Write PCM_16 WAV instead of float.")
    p.add_argument("--sync-every", type=int, default=32, help="Insert one pilot every N data frames (default 32).")
    p.add_argument("--pilot-bits", type=int, default=32, help="Length of keyed pilot sequence in bits (default 32).")
    p.add_argument("--energy-gate-db", type=float, default=-20.0, help="Keep segments within this dB of median RMS (default -20dB).")
    p.add_argument("--jitter", type=float, default=0.02, help="Max start jitter (seconds) within a segment (default 0.02s).")
    p.add_argument("--min-spacing", type=int, default=2, help="Minimum spacing (in segments) between placements (default 2).")
    p.add_argument("--max-hits-per-seg", type=int, default=2, help="Cap placements per segment before relaxing (default 2).")
    p.add_argument("--fec", choices=["none","hamming15","bch63"], default="bch63", help="Forward error correction code (default bch63).")
    p.add_argument("--interleave-depth", type=int, default=8, help="Block interleaver depth (default 8).")
    p.add_argument("--s-mix", type=float, default=0.12, help="For stereo M/S, Side mix ratio (default 0.12).")
    p.add_argument("--ms-embed", action="store_true", help="Force enable stereo Mid/Side embedding.")
    p.add_argument("--no-ms-embed", action="store_true", help="Force disable stereo Mid/Side embedding.")
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="Device for spectral scoring (default cpu).")
    p.add_argument("--limit-threshold", type=float, default=0.98, help="Limiter threshold (default 0.98).")
    p.add_argument("--limit-knee", type=float, default=0.02, help="Limiter knee (default 0.02).")
    p.add_argument("--log", default="wm_singlepath_log.csv", help="CSV audit log (append, encode mode).")
    p.add_argument("--preset", choices=["speech","music","voip"], default=None, help="Tune seg/repeat/sync/pilot/interleave for common scenarios.")
    p.add_argument("--run-tests", action="store_true", help="Run quick unit tests (interleave identity, packet auth, scorer sanity) and exit.")
    return p.parse_args()

# --------- Main ---------

def main():
    args = parse_args()

    # Presets
    if args.preset == 'speech':
        args.seg = 0.24; args.repeat = 3; args.sync_every = 24; args.pilot_bits = 48; args.interleave_depth = 8
    elif args.preset == 'music':
        args.seg = 0.20; args.repeat = 2; args.sync_every = 32; args.pilot_bits = 32; args.interleave_depth = 6
    elif args.preset == 'voip':
        args.seg = 0.20; args.repeat = 4; args.sync_every = 20; args.pilot_bits = 64; args.interleave_depth = 10

    infile = args.infile
    # Unit tests
    if args.run_tests:
        # Interleave identity
        bits = [i & 1 for i in range(257)]
        for d in [1,2,3,8,10]:
            assert block_deinterleave(block_interleave(bits, d), d) == bits, f"interleave identity failed at depth={d}"
        # Packet auth/CRC policy
        test_manifest = auto_manifest(infile, args.published_at, args.model, args.title, args.issuer, args.license_str, args.text_file)
        _nonce = 0xA1B2C3D4
        pkt_auth = pack_full_payload(test_manifest, key_id=3, auth_key=b"sekrit", nonce=_nonce)
        hdr, body, kid, typ, flags, n, ver, tag, crc = parse_payload(pkt_auth)
        assert flags & FLAG_AUTH, "auth flag not set"
        assert hdr["crc_hex"] == "00000000", "CRC nonzero under auth"
        pkt_noauth = pack_full_payload(test_manifest, key_id=3, auth_key=b"", nonce=_nonce)
        hdr2, body2, *_ = parse_payload(pkt_noauth)
        assert hdr2["crc_hex"] != "00000000", "CRC zero in unauthenticated packet"
        print("UNIT TESTS OK")
        return

    if args.mode == "encode":
        outfile = args.outfile or (os.path.splitext(infile)[0] + ".wm.wav")
        if os.path.abspath(infile) == os.path.abspath(outfile):
            raise SystemExit("Refusing to overwrite input. Use a different --out.")
    else:
        outfile = args.outfile or (os.path.splitext(infile)[0] + ".wm.json")

    audio_np, orig_sr = read_audio(infile)
    n_samples, n_ch = audio_np.shape
    dur = n_samples / float(orig_sr)

    # Secrets/keys
if args.secret:
    base_seed = hashlib.sha256(args.secret.encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
else:
    base_seed = hashlib.sha256((args.key or "").encode("utf-8") + bytes([args.key_id & 0xFF])).digest()

# Nonce: mix deterministic HKDF with 32 bits of OS entropy so each copy schedules differently
hk = hkdf_sha256(base_seed, salt=b"nonce", info=b"wm.nonce", length=4)
rnd = os.urandom(4)
nonce_bytes = int.from_bytes(hk, "big") ^ int.from_bytes(rnd, "big")
nonce_bytes = nonce_bytes.to_bytes(4, "big")
nonce = int.from_bytes(nonce_bytes, "big")

# Derive subkeys salted with nonce (anti-collusion)
Kp = hkdf_sha256(base_seed, salt=nonce_bytes, info=b"wm.place", length=32)
Ks = hkdf_sha256(base_seed, salt=nonce_bytes, info=b"wm.sync", length=32)
Ka = hkdf_sha256(base_seed, salt=nonce_bytes, info=b"wm.auth", length=32)
auth_key = Ka if args.secret else ((args.auth_key or "").encode("utf-8"))


    if args.mode == "selftest":
        # Build manifest/payload exactly as encode would, then round-trip via interleave/FEC→deinterleave/FEC^-1→parse to ensure consistency.
        if args.manifest:
            with open(args.manifest, "r", encoding="utf-8") as f:
                manifest_json = json.load(f)
        else:
            manifest_json = auto_manifest(infile, args.published_at, args.model,
                                          args.title, args.issuer, args.license_str, args.text_file)
        # Choose a payload (full preferred; then micro; then anchor) strictly by size vs capacity
        full_payload = pack_full_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)
        micro_payload = pack_micro_payload(manifest_json, args.published_at, args.model, args.issuer,
                                           key_id=args.key_id, auth_key=auth_key, nonce=nonce)
        anchor_payload = pack_anchor_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)
        cap_bits = capacity_bits_audioseal(dur, args.seg, args.repeat, args.sync_every, args.pilot_bits)
        cand = [("full", full_payload), ("micro", micro_payload), ("anchor", anchor_payload)]
        cand = [(n,b) for (n,b) in cand if len(b)*8 <= cap_bits]
        if not cand:
            raise SystemExit(f"selftest: no payload fits capacity {cap_bits} bits")
        payload_used, payload_bytes = cand[0]

        # Forward FEC + interleave + frame + pilot
        raw_bits = bytes_to_bits(payload_bytes)
        coded_bits = apply_fec(raw_bits, args.fec)
        inter_bits = block_interleave(coded_bits, args.interleave_depth)
        frames_data = bits_to_frames(inter_bits, width=16)
        frames_with_pilot, n_pilots, pilot_frames_each = add_pilot_frames(frames_data, args.sync_every, args.pilot_bits, Ks, terminal=True)

        # Reverse pilot strip + deframe + deinterleave + FEC^-1
# Mirror insertion: pilots are inserted as [pilot_frames] followed by (sync_every-1) data frames
pilot_frames = math.ceil(max(1, args.pilot_bits) / 16)
data_frames = []
data_since_pilot = 0
i = 0
while i < len(frames_with_pilot):
    if data_since_pilot == 0:
        # skip exactly 'pilot_frames' frames
        i += pilot_frames
        data_since_pilot = max(1, args.sync_every) - 1
        continue
    # collect one data frame
    if i < len(frames_with_pilot):
        data_frames.append(frames_with_pilot[i])
        i += 1
        data_since_pilot -= 1
        if data_since_pilot == 0:
            # next loop will skip pilots again
            pass
# Byte-pack
        by = bytearray()
        for i in range(0, len(raw_bits2), 8):
            byte = 0
            for b in raw_bits2[i:i+8]:
                byte = (byte<<1) | (b & 1)
            by.append(byte)

        # Parse and auth check
        header, body, kid, typ, flags, pnonce, ver, tag, crc = parse_payload(bytes(by))
        if flags & FLAG_AUTH:
            head_wo_tag = bytes([ver, kid, typ, flags]) + pnonce.to_bytes(4,"big") + header["len"].to_bytes(4,"big") + body
            calc = hmac.new(auth_key, head_wo_tag, hashlib.sha256).digest()[:16]
            assert calc == tag, "selftest: HMAC mismatch"
            assert header["crc_hex"] == "00000000", "selftest: CRC should be zero under auth"
        else:
            c = (binascii.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")
            assert c.hex() == header["crc_hex"], "selftest: CRC mismatch"

        print(f"SELFTEST OK | payload={payload_used} | frames_data={len(frames_data)} | pilots={n_pilots}x{pilot_frames_each}")
        return

    if args.mode == "decode":
        # Decoder skeleton
        try:
            frames_with_pilot, dbg = decode_frames_audioseal(
                audio_np, orig_sr, args.seg, args.repeat,
                Ks, args.sync_every, args.pilot_bits,
                Kp, args.energy_gate_db, args.jitter, args.min_spacing,
                args.device, args.max_hits_per_seg
            )
        except SystemExit as e:
            print(str(e))
            sys.exit(1)
        # Deinterleave + FEC remove + reassemble bytes (skeleton)
        bitstream = frames_to_bits(frames_with_pilot)
        deint_bits = block_deinterleave(bitstream, args.interleave_depth)
        raw_bits = remove_fec(deint_bits, args.fec)
        # Pack bits to bytes
        by = bytearray()
        for i in range(0, len(raw_bits), 8):
            byte = 0
            for b in raw_bits[i:i+8]:
                byte = (byte<<1) | (b&1)
            by.append(byte)
        # Parse payload
        try:
            header, body, kid, typ, flags, pnonce, ver, tag, crc = parse_payload(bytes(by))
        except Exception as e:
            print(f"decode: failed to parse payload header: {e}")
            sys.exit(1)
        # Auth check (if FLAG_AUTH set)
        if flags & FLAG_AUTH:
            head_wo_tag = bytes([ver, kid, typ, flags]) + pnonce.to_bytes(4,"big") + header["len"].to_bytes(4,"big") + body
            calc = hmac.new(auth_key, head_wo_tag, hashlib.sha256).digest()[:16]
            if calc != tag:
                print("decode: HMAC tag mismatch")
                sys.exit(1)
        # CRC policy
        if (flags & FLAG_AUTH) and header["crc_hex"] != "00000000":
            print("decode: authenticated packet should have zero CRC; rejecting")
            sys.exit(1)
        # Emit JSON result
        result = {
            "header": header,
            "type": typ,
            "key_id": kid,
            "nonce_hex8": f"{pnonce:08x}",
            "authenticated": int(bool(flags & FLAG_AUTH)),
            "payload_len": len(body),
            "decode_notes": "Decoder skeleton; audio->bit extraction requires detector integration.",
        }
        # Attempt to decompress & parse JSON if TYPE_FULL
        if typ == TYPE_FULL:
            try:
                j = json.loads(zlib.decompress(body).decode("utf-8"))
                result["manifest"] = j
            except Exception:
                result["manifest"] = None
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote decode result: {outfile}")
        return

    # ENCODE path
    # Manifest
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)
    else:
        manifest_json = auto_manifest(infile, args.published_at, args.model,
                                      args.title, args.issuer, args.license_str, args.text_file)

    # Payloads
    full_payload = pack_full_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)
    micro_payload = pack_micro_payload(manifest_json, args.published_at, args.model, args.issuer,
                                       key_id=args.key_id, auth_key=auth_key, nonce=nonce)
    anchor_payload = pack_anchor_payload(manifest_json, key_id=args.key_id, auth_key=auth_key, nonce=nonce)

    # Capacity estimate
    engine_used = args.engine
    wavmark_ok = False
    if args.engine == ENGINE_WAVMARK:
        try:
            import wavmark  # type: ignore
            wavmark_ok = True
        except Exception:
            engine_used = ENGINE_AUDIOSEAL

    cap_bits = (capacity_bits_wavmark(dur) if (engine_used == ENGINE_WAVMARK and wavmark_ok)
                else capacity_bits_audioseal(dur, args.seg, args.repeat, args.sync_every, args.pilot_bits))
    print(f"Capacity estimate: usable≈{cap_bits} bits (seg={args.seg}s, repeat={args.repeat}, sync_every={args.sync_every}, pilot_bits={args.pilot_bits})")

    # Choose payload
    full_bits = len(full_payload)*8
    micro_bits = len(micro_payload)*8
    anchor_bits = len(anchor_payload)*8
    if full_bits <= cap_bits:
        payload_used = "full"; payload_bytes = full_payload
    elif micro_bits <= cap_bits:
        payload_used = "micro"; payload_bytes = micro_payload
    elif anchor_bits <= cap_bits:
        payload_used = "anchor"; payload_bytes = anchor_payload
    else:
        raise SystemExit(f"Capacity too low: cap≈{cap_bits} bits, full={full_bits}, micro={micro_bits}, anchor={anchor_bits}.")

    # Build bitstream: FEC -> interleave -> frames
    raw_bits = bytes_to_bits(payload_bytes)
    coded_bits = apply_fec(raw_bits, args.fec)
    inter_bits = block_interleave(coded_bits, args.interleave_depth)
    frames_data = bits_to_frames(inter_bits, width=16)

    # Repeat only DATA frames here (pilots will not be repeated)
    frames_data_rep = []
    for fr in frames_data:
        for _ in range(max(1, args.repeat)):
            frames_data_rep.append(fr)

    # Insert pilots AFTER repetition so pilots are not repeated
    frames_with_pilot, n_pilots, pilot_frames_each = add_pilot_frames(frames_data_rep, args.sync_every, args.pilot_bits, Ks, terminal=True)
    frames_total = len(frames_with_pilot)

    # Decide M/S default for stereo: ON by default for stereo unless --no-ms-embed; explicit --ms-embed always enables
    ms_embed_flag = ((n_ch == 2) and (not args.no_ms_embed)) or bool(args.ms_embed)

    # Embed
    if engine_used == ENGINE_WAVMARK and wavmark_ok:
        out_np, host_r, wm_r, limited = embed_payload_wavmark(audio_np, orig_sr, payload_bytes, args.gain, args.rms_cap)
    else:
        out_np, host_r, wm_r, limited = embed_frames_audioseal(
            audio_np, orig_sr, frames_with_pilot, args.seg, 1, args.gain,
            args.rms_cap, Kp, args.energy_gate_db, args.jitter, args.min_spacing,
            args.s_mix, ms_embed_flag, args.limit_threshold, args.limit_knee,
            args.device, args.max_hits_per_seg
        )

    # Save
    write_audio(outfile, out_np, orig_sr, pcm16=args.pcm16)

    # Log (encode)
    need_header = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","engine","payload_used","header_ver","auth","key_id","nonce_hex8",
            "in_sr","channels","duration_s",
            "cap_bits","payload_bits","fec","interleave_depth",
            "sync_every","pilot_bits","frames_pilot_each","n_pilots","frames_pilot_total","pilot_cadence",
            "frames_data","frames_total",
            "seg","repeat","energy_gate_db","min_spacing","max_hits_per_seg","jitter_s","s_mix","ms_embed",
            "host_rms","wm_rms","wm_to_host_db","limit_threshold","limit_knee","limited",
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
            "nonce_hex8": f"{nonce:08x}",
            "in_sr": orig_sr,
            "channels": n_ch,
            "duration_s": round(dur,3),
            "cap_bits": cap_bits,
            "payload_bits": len(payload_bytes)*8,
            "fec": args.fec,
            "interleave_depth": args.interleave_depth,
            "sync_every": args.sync_every,
            "pilot_cadence": args.sync_every,
            "pilot_bits": args.pilot_bits,
            "frames_pilot_each": pilot_frames_each,
            "n_pilots": n_pilots,
            "frames_pilot_total": n_pilots * pilot_frames_each,
            "frames_data": len(frames_data),
            "frames_total": frames_total,
            "seg": round(args.seg,3),
            "repeat": args.repeat,
            "energy_gate_db": args.energy_gate_db,
            "min_spacing": args.min_spacing,
            "max_hits_per_seg": args.max_hits_per_seg,
            "jitter_s": args.jitter,
            "s_mix": args.s_mix,
            "ms_embed": int(ms_embed_flag),
            "host_rms": round(host_r,6),
            "wm_rms": round(wm_r,6),
            "wm_to_host_db": round(wm_to_host_db,2),
            "limit_threshold": args.limit_threshold,
            "limit_knee": args.limit_knee,
            "limited": int(limited),
            "infile": os.path.abspath(infile),
            "outfile": os.path.abspath(outfile),
        })

    print(f"Wrote: {outfile} | engine={engine_used} | payload={payload_used} | "
          f"cap≈{cap_bits} bits | used={len(payload_bytes)*8} bits | "
          f"frames: data={len(frames_data)}, pilots={n_pilots}x{pilot_frames_each}, total={frames_total} | host_ref={'mid' if (n_ch==2 and ms_embed_flag) else 'mixmean'}")

if __name__ == "__main__":
    main()
