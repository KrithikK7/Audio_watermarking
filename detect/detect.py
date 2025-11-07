#!/usr/bin/env python3
# detect.py — Standalone decoder for the single‑path audio watermark (v6-compatible)
# - Extracts 16-bit frames with AudioSeal detector
# - Mirrors cadence: [pilot_frames] + (sync_every-1) data, repeated, plus terminal pilot
# - Strips pilots, folds repeats, deinterleaves, FEC^-1, parses header, verifies HMAC/CRC
#
# Requires: audioseal, torchaudio, torch, soundfile; optional: bchlib for BCH(63,45)

import argparse, json, math, os, sys, binascii, hmac, hashlib, struct, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import soundfile as sf
from torchaudio.transforms import Resample

# ---- AudioSeal ----
try:
    from audioseal import AudioSeal
except Exception as e:
    print("FATAL: audioseal package not available:", e)
    sys.exit(1)

AS_SR = 16000
AS_MODEL_NAME = "audioseal_detector_16bits"
MIN_SEG_SEC = 0.16  # floor for segment duration

# ---- HKDF (SHA256) ----
def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

# ---- Bits / frames / interleaver ----
def bytes_to_bits(by: bytes) -> List[int]:
    out = []
    for b in by:
        for k in range(7, -1, -1):
            out.append((b >> k) & 1)
    return out

def frames_to_bits(frames: List[List[int]]) -> List[int]:
    out = []
    for fr in frames:
        out.extend(fr[:16])
    return out

def bits_to_frames(bits: List[int], width: int = 16) -> List[List[int]]:
    out = []
    i = 0
    while i < len(bits):
        chunk = bits[i:i+width]
        if len(chunk) < width:
            chunk += [0] * (width - len(chunk))
        out.append(chunk)
        i += width
    return out

def block_interleave(bits: List[int], depth: int) -> List[int]:
    if depth <= 1:
        return bits
    n = len(bits)
    cols = depth
    rows = math.ceil(n / cols)
    padded = bits + [0] * (rows * cols - n)
    out = []
    for c in range(cols):
        for r in range(rows):
            out.append(padded[r * cols + c])
    return out[:n]

def block_deinterleave(bits: List[int], depth: int) -> List[int]:
    if depth <= 1:
        return bits
    n = len(bits)
    cols = depth
    rows = math.ceil(n / cols)
    padded = bits + [0] * (rows * cols - n)
    out = [0] * (rows * cols)
    idx = 0
    for c in range(cols):
        for r in range(rows):
            out[r * cols + c] = padded[idx]
            idx += 1
    return out[:n]

# ---- Simple Hamming(15,11) ----
def hamming15_decode_bits(bits: List[int]) -> List[int]:
    out: List[int] = []
    i = 0
    while i + 15 <= len(bits):
        cw = [None] + bits[i:i+15]  # 1-indexed
        def parity(vals):
            s = 0
            for v in vals: s ^= (v & 1)
            return s & 1
        p1 = parity([cw[j] for j in [1,3,5,7,9,11,13,15]])
        p2 = parity([cw[j] for j in [2,3,6,7,10,11,14,15]])
        p4 = parity([cw[j] for j in [4,5,6,7,12,13,14,15]])
        p8 = parity([cw[j] for j in [8,9,10,11,12,13,14,15]])
        s = (p8<<3)|(p4<<2)|(p2<<1)|p1
        if s != 0 and 1 <= s <= 15: cw[s] ^= 1
        data_pos = [3,5,6,7,9,10,11,12,13,14,15]
        for dp in data_pos: out.append(cw[dp] & 1)
        i += 15
    return out

# ---- BCH(63,45) byte-true (optional) ----
def bch63_remove(bits: List[int]) -> List[int]:
    try:
        import bchlib  # type: ignore
    except Exception:
        raise SystemExit("FEC 'bch63' requires the 'bchlib' package. Install it or choose --fec hamming15/none.")
    BCH_POLY = 8219
    bch = bchlib.BCH(BCH_POLY, t=3)
    DATA_LEN = 6
    CODEWORD_BYTES = DATA_LEN + bch.ecc_bytes

    # bits -> bytes
    byt = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8: chunk += [0]*(8-len(chunk))
        v = 0
        for b in chunk: v = (v<<1) | (b & 1)
        byt.append(v & 0xFF)

    out_bits: List[int] = []
    pos = 0
    while pos + CODEWORD_BYTES <= len(byt):
        packet = byt[pos:pos+CODEWORD_BYTES]
        data = bytes(packet[:DATA_LEN]); ecc = bytes(packet[DATA_LEN:])
        d_corr, ecc_corr = bch.decode(bytearray(data), bytearray(ecc))
        val = int.from_bytes(d_corr, "big")
        start_bit = DATA_LEN*8 - 1
        for j in range(45):
            bit = (val >> (start_bit - j)) & 1
            out_bits.append(bit)
        pos += CODEWORD_BYTES
    return out_bits

def remove_fec(bits: List[int], fec: str) -> List[int]:
    if fec == "none":
        return bits
    if fec == "hamming15":
        return hamming15_decode_bits(bits)
    if fec == "bch63":
        return bch63_remove(bits)
    raise SystemExit(f"Unknown FEC '{fec}'")

# ---- Pilot bits (v6: derived only from secret; nonce not mixed) ----
def derive_pilot_bits(Ks: bytes, pilot_bits: int) -> List[int]:
    need = max(1, pilot_bits)
    out = []
    counter = 0
    while len(out) < need:
        block = hmac.new(Ks, f"pilot-{counter}".encode(), hashlib.sha256).digest()
        for b in block:
            for k in range(7,-1,-1):
                out.append((b >> k) & 1)
                if len(out) >= need:
                    break
            if len(out) >= need:
                break
        counter += 1
    return out

# ---- Energy gating + spectral scoring (1–4 kHz emphasis) ----
def energy_gated_indices(x_m: torch.Tensor, seg_len: int, gate_db: float) -> List[int]:
    if x_m.numel() < seg_len: return []
    nseg = x_m.numel() // seg_len
    rms = []
    for i in range(nseg):
        s = i*seg_len; e = s + seg_len
        seg = x_m[s:e]
        r = torch.sqrt(torch.clamp((seg**2).mean(), 1e-12))
        rms.append(float(r))
    med = np.median(np.array(rms)) if rms else 0.0
    if med <= 0: return list(range(nseg))
    thr = med * (10.0 ** (gate_db / 20.0))  # gate_db is negative (e.g., -10 dB)
    pool = [i for i, r in enumerate(rms) if r >= thr]
    return pool if pool else list(range(nseg))

def spectral_scores(x_m: torch.Tensor, seg_len: int, sr: int = AS_SR) -> np.ndarray:
    nseg = x_m.numel() // seg_len
    if nseg == 0: return np.zeros(0, dtype=np.float32)
    win = torch.hann_window(seg_len, periodic=True)
    scores = np.zeros(nseg, dtype=np.float32)
    f = torch.fft.rfft(win)  # not used; placeholder to warm FFT
    for i in range(nseg):
        s = i*seg_len; e = s + seg_len
        seg = x_m[s:e] * win
        spec = torch.fft.rfft(seg)
        mag = (spec.real**2 + spec.imag**2).sqrt()
        freqs = torch.fft.rfftfreq(seg_len, 1.0/sr)
        band = ((freqs >= 1000.0) & (freqs <= 4000.0)).float()
        score = float((mag * band).sum() / (mag.sum() + 1e-9))
        scores[i] = score
    return scores

# ---- Schedule (seeded by placement key) ----
def schedule_positions(pool: List[int], total_needed: int, min_spacing: int,
                       seed: bytes, scores: np.ndarray,
                       max_hits_per_seg: int, n_segments: int) -> List[int]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "big"))
    # base candidates sorted by score desc, then random tie-break
    base = list(pool)
    if scores.size:
        base.sort(key=lambda i: (-scores[i], rng.random()))
    else:
        rng.shuffle(base)
    # expand with wrap-around if needed
    if len(base) < total_needed:
        reps = math.ceil(total_needed / max(1, len(base)))
        base = (base * reps)[:total_needed]
    chosen = []
    hits = {}
    idx = 0
    while len(chosen) < total_needed and idx < len(base):
        cand = base[idx]; idx += 1
        if hits.get(cand, 0) >= max_hits_per_seg: continue
        if chosen and min(abs(cand - c) for c in chosen) < max(1, min_spacing): continue
        chosen.append(cand); hits[cand] = hits.get(cand, 0) + 1
    # Greedy farthest-point reorder to spread locally
    if not chosen: return chosen
    pts = sorted(set(chosen))
    def mean_dist(i): return sum(abs(i - j) for j in pts) / max(1, len(pts)-1)
    seed_pt = max(pts, key=mean_dist)
    ordered = [seed_pt]
    rem = [i for i in pts if i != seed_pt]
    while rem:
        best = None; best_d = -1
        for c in rem:
            d = min(abs(c - o) for o in ordered)
            if d > best_d: best_d = d; best = c
        ordered.append(best); rem.remove(best)
    # Map back to multiplicity of chosen
    # Build counts
    counts = {i: chosen.count(i) for i in set(chosen)}
    out = []
    for o in ordered:
        k = counts[o]
        out.extend([o]*k)
    # If still short, fill random
    while len(out) < total_needed:
        out.append(rng.integers(0, n_segments))
    return out[:total_needed]

# ---- Helpers: pilots & repeats ----
def strip_pilot_frames(frames_with_pilot: List[List[int]], sync_every: int, pilot_bits: int) -> List[List[int]]:
    pilot_frames = max(1, (pilot_bits + 15) // 16)
    out = []
    i = 0; n = len(frames_with_pilot)
    while i < n:
        i += pilot_frames
        for _ in range(max(1, sync_every) - 1):
            if i >= n: break
            out.append(frames_with_pilot[i])
            i += 1
    return out

def fold_repeats(data_frames: List[List[int]], repeat: int) -> List[List[int]]:
    if repeat <= 1: return data_frames
    out = []
    for i in range(0, len(data_frames), repeat):
        grp = data_frames[i:i+repeat]
        if len(grp) < repeat: break
        sums = [sum(fr[j] for fr in grp) for j in range(16)]
        bits = [1 if (2*s) >= repeat else 0 for s in sums]
        out.append(bits)
    return out

# ---- Payload parse (v6 header) ----
FLAG_AUTH = 0x01

def parse_payload(by: bytes):
    if len(by) < 1 + 1 + 1 + 1 + 4 + 4 + 16 + 4:
        raise SystemExit("decode: payload too short")
    ver, kid, typ, flags = by[0], by[1], by[2], by[3]
    nonce = int.from_bytes(by[4:8], "big")
    body_len = int.from_bytes(by[8:12], "big")
    hmac16 = by[12:28]
    crc = by[28:32]
    body = by[32:32+body_len]
    if len(body) != body_len:
        raise SystemExit("decode: truncated payload body")
    header = {
        "ver": ver, "key_id": kid, "type": typ, "flags": flags,
        "nonce": nonce, "len": body_len,
        "hmac16_hex": hmac16.hex(), "crc_hex": crc.hex()
    }
    return header, body, kid, typ, flags, nonce, ver, hmac16, crc

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Standalone decoder for single-path audio watermark (v6).")
    p.add_argument("--in", dest="infile", required=True, help="Input WAV (watermarked).")
    p.add_argument("--secret", required=True, help="Decode secret (passphrase).")
    p.add_argument("--key-id", type=int, default=1, help="Key id (0..255).")
    p.add_argument("--seg", type=float, default=0.24, help="Segment seconds (must match encoder).")
    p.add_argument("--repeat", type=int, default=3, help="Repeat count (data frames).")
    p.add_argument("--sync-every", type=int, default=24, help="Pilot cadence (insert every N frames).")
    p.add_argument("--pilot-bits", type=int, default=48, help="Pilot bit length.")
    p.add_argument("--interleave-depth", type=int, default=8, help="Block interleaver depth.")
    p.add_argument("--min-spacing", type=int, default=2, help="Min segment spacing (in segments).")
    p.add_argument("--max-hits-per-seg", type=int, default=2, help="Max placements per segment.")
    p.add_argument("--energy-gate-db", type=float, default=-10.0, help="Energy gate (dB rel. median).")
    p.add_argument("--fec", choices=["none","hamming15","bch63"], default="bch63", help="FEC used at encode.")
    p.add_argument("--device", default="cpu", help="PyTorch device for scoring (cpu/cuda).")
    p.add_argument("--json", default=None, help="Output JSON path (default: <in>.wm.json).")
    return p.parse_args()

def main():
    args = parse_args()
    inpath = Path(args.infile)

    # ---- Load audio ----
    wav, sr = sf.read(str(inpath), always_2d=True)
    x = torch.from_numpy(wav).float()
    if x.shape[1] >= 2:
        mid = (x[:,0] + x[:,1]) * 0.5
    else:
        mid = x[:,0]
    if sr != AS_SR:
        mid = Resample(orig_freq=sr, new_freq=AS_SR)(mid.unsqueeze(0)).squeeze(0)

    seg_len = int(max(MIN_SEG_SEC, args.seg) * AS_SR)
    nseg = max(1, mid.numel() // seg_len)

    # ---- Derive subkeys (v6: no nonce mixing) ----
    base_seed = hashlib.sha256(args.secret.encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
    Kp = hkdf_sha256(base_seed, salt=b"place", info=b"wm.place", length=32)
    Ks = hkdf_sha256(base_seed, salt=b"sync", info=b"wm.sync", length=32)
    Ka = hkdf_sha256(base_seed, salt=b"auth", info=b"wm.auth", length=32)
    auth_key = Ka  # HMAC derivation

    # ---- Estimate schedule length (must match encoder logic) ----
    # frames_capacity = number of positions for repeated data frames (each repetition consumes a position)
    frames_capacity = nseg // max(1, args.repeat)
    pilot_frames_each = max(1, (args.pilot_bits + 15) // 16)
    # one pilot per cadence slice + a terminal pilot
    n_pilots_total = (max(1, frames_capacity * args.repeat) // max(1, args.sync_every)) + 1
    frames_total = frames_capacity * args.repeat + n_pilots_total * pilot_frames_each

    # ---- Build positions ----
    pool = energy_gated_indices(mid, seg_len, gate_db=args.energy_gate_db)
    scores = spectral_scores(mid, seg_len, sr=AS_SR)
    positions = schedule_positions(pool, frames_total, args.min_spacing, seed=Kp,
                                   scores=scores, max_hits_per_seg=args.max_hits_per_seg, n_segments=nseg)

    # ---- Detector ----
    det = AudioSeal.load_detector(AS_MODEL_NAME)

    # Pilot frames (known bits)
    pbits = derive_pilot_bits(Ks, max(1, args.pilot_bits))
    pilot_frames = frames = bits_to_frames(pbits, width=16)

    frames_with_pilot = []
    idx = 0
    total_positions = len(positions)

    # Mirror cadence: [pilot_frames] + (sync_every-1) data, repeated; append terminal pilot
    while idx < total_positions:
        # pilot(s)
        for pf in pilot_frames:
            frames_with_pilot.append([int(b) & 1 for b in pf])
            idx += 1
            if idx >= total_positions:
                break
        if idx >= total_positions:
            break
        # data
        for _ in range(max(1, args.sync_every) - 1):
            if idx >= total_positions: break
            seg_idx = positions[idx]
            s = seg_idx * seg_len; e = min(s + seg_len, mid.numel())
            seg = mid[s:e]
            if seg.numel() < seg_len:
                seg = torch.nn.functional.pad(seg, (0, seg_len - seg.numel()))
            try:
                vb = det.get_bits(seg, sample_rate=AS_SR)
                bits = [int(b) & 1 for b in vb][:16]
            except Exception:
                logits = det.get_logits(seg, sample_rate=AS_SR)
                arr = list(logits) if hasattr(logits, "__iter__") else [float(logits)]
                if len(arr) < 16: arr += [0.0] * (16 - len(arr))
                bits = [1 if x > 0 else 0 for x in arr[:16]]
            frames_with_pilot.append(bits)
            idx += 1

    # Fill remainder with terminal pilot
    while idx < total_positions:
        for pf in pilot_frames:
            frames_with_pilot.append([int(b) & 1 for b in pf])
            idx += 1
            if idx >= total_positions: break

    # ---- Strip pilots, fold repeats, assemble ----
    data_frames = strip_pilot_frames(frames_with_pilot, args.sync_every, args.pilot_bits)
    data_frames = fold_repeats(data_frames, args.repeat)
    bitstream = frames_to_bits(data_frames)
    deint = block_deinterleave(bitstream, args.interleave_depth)
    raw_bits = remove_fec(deint, args.fec)

    # pack bits -> bytes
    by = bytearray()
    for i in range(0, len(raw_bits), 8):
        v = 0
        for b in raw_bits[i:i+8]:
            v = (v << 1) | (b & 1)
        by.append(v & 0xFF)

    # ---- Parse + verify ----
    header, body, kid, typ, flags, nonce, ver, hmac16, crc = parse_payload(bytes(by))
    # HMAC policy (v6): HMAC over [ver,kid,typ,flags,nonce,len,body], truncated 16
    head_wo_tag = bytes([ver, kid, typ, flags]) + nonce.to_bytes(4, "big") + header["len"].to_bytes(4, "big") + body
    if flags & 0x01:
        calc = hmac.new(auth_key, head_wo_tag, hashlib.sha256).digest()[:16]
        if calc != hmac16:
            raise SystemExit("decode: HMAC verification failed")
        crc_hex = "00000000"
    else:
        crc_hex = (binascii.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big").hex()
        if crc_hex != header["crc_hex"]:
            raise SystemExit("decode: CRC mismatch (unauthenticated)")

    out = {
        "header": header,
        "payload_type": typ,
        "kid": kid,
        "nonce": nonce,
        "len": header["len"],
        "auth": bool(flags & 0x01),
        "crc_ok": (crc_hex == "00000000") if (flags & 0x01) else True,
    }
    out_path = Path(args.json) if args.json else Path(args.infile).with_suffix(".wm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("DECODE OK →", str(out_path))

if __name__ == "__main__":
    main()