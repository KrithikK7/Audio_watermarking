#!/usr/bin/env python3
"""
detect.py — Detector for wm_singlepath (v7)
Default outputs and CSVs live in:
    B:\Krithik\Project\Audio_watermarking\Logs
"""

import argparse, json, os, sys, zlib, hashlib, hmac, csv, io, math
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import soundfile as sf
try:
    from torchaudio.transforms import Resample
except Exception:
    Resample = None

# --------- FIXED LOG/OUTPUT DIRECTORY ---------
LOG_DIR = os.path.normpath(r"B:\Krithik\Project\Audio_watermarking\Logs")
os.makedirs(LOG_DIR, exist_ok=True)
def log_path(*parts: str) -> str:
    return os.path.abspath(os.path.join(LOG_DIR, *parts))

# ---- Prefer helpers from wm_singlepath; fallback to embed.py ----
try:
    from wm_singlepath import (
        AS_SR, MIN_SEG_SEC,
        hkdf_sha256, derive_pilot_bits, bits_to_frames, frames_to_bits,
        block_deinterleave, remove_fec, parse_payload,
        SpectralScorer, energy_gated_indices, schedule_positions, ensure_len,
    )
except Exception:
    from embed import (
        AS_SR, MIN_SEG_SEC,
        hkdf_sha256, derive_pilot_bits, bits_to_frames, frames_to_bits,
        block_deinterleave, remove_fec, parse_payload,
        SpectralScorer, energy_gated_indices, schedule_positions, ensure_len,
    )

DET_MODEL_NAME = "audioseal_detector_16bits"

# --------- I/O ---------
def read_audio(path: str):
    x, sr = sf.read(path, always_2d=True)
    return x.astype(np.float32, copy=False), sr

def resample_mono(x: np.ndarray, sr_in: int) -> torch.Tensor:
    mono = x.mean(axis=1).astype(np.float32, copy=False)
    t = torch.from_numpy(mono)
    if sr_in == AS_SR:
        return t
    if Resample is None:
        ratio = AS_SR / float(sr_in)
        n_out = int(math.floor(len(t) * ratio))
        if n_out <= 1:
            return t
        idx = torch.linspace(0, len(t)-1, n_out, dtype=torch.float32)
        idx0 = torch.clamp(idx.floor().long(), 0, len(t)-1)
        idx1 = torch.clamp(idx0+1, 0, len(t)-1)
        frac = idx - idx0.float()
        return (t[idx0] * (1-frac) + t[idx1] * frac)
    return Resample(orig_freq=sr_in, new_freq=AS_SR)(t.unsqueeze(0)).squeeze(0)

# --------- Utils ---------
def hamming(a: List[int], b: List[int]) -> int:
    return sum((int(x) ^ int(y)) & 1 for x, y in zip(a, b))

def majority_bits(frames: List[List[int]]) -> List[int]:
    if not frames: return []
    w = len(frames[0]); out = []
    for i in range(w):
        s = sum(fr[i] for fr in frames)
        out.append(1 if s * 2 >= len(frames) else 0)
    return out

def pack_bits_to_bytes(bits: List[int]) -> bytes:
    by = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (b & 1)
        by.append(byte & 0xFF)
    return bytes(by)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def audio_meta(path: str) -> tuple:
    info = sf.info(path)
    return (int(info.samplerate), int(info.channels), round(float(info.duration), 3))

def normpath_ci(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def slice_to_csv_from_header(text: str) -> str:
    idx = text.find("timestamp,")
    return text[idx:] if idx >= 0 else text

# --------- Embed/Detect CSV paths (fixed) ---------
def default_embed_csv() -> str:
    return log_path("wm_singlepath_log.csv")

def default_detect_csv() -> str:
    return log_path("wm_detect_log.csv")

def nonce_index_csv() -> str:
    return log_path("nonce_index.csv")

def estimate_frame_requirements(row: dict, mono_len: int, seg_len_m: int, args) -> tuple:
    frames_total = int(row.get("frames_total") or 0) if row else 0
    frames_pilot_each = int(row.get("frames_pilot_each") or 0) if row else 0
    n_pilots = int(row.get("n_pilots") or 0) if row else 0
    pilot_bits_row = int(row.get("pilot_bits") or 0) if row else 0
    pilot_bits = pilot_bits_row or int(getattr(args, "pilot_bits", 0) or 0)
    if frames_pilot_each <= 0:
        frames_pilot_each = max(1, math.ceil(max(1, pilot_bits) / 16))
    if frames_total <= 0 or n_pilots <= 0:
        n_segments = max(1, mono_len // max(1, seg_len_m))
        frames_capacity = n_segments // max(1, getattr(args, "repeat", 1))
        n_pilots = (frames_capacity // max(1, getattr(args, "sync_every", 1))) + 1
        frames_total = frames_capacity + n_pilots * frames_pilot_each
    min_frames = max(frames_pilot_each * max(1, n_pilots), frames_total // 4, 8)
    return frames_total, frames_pilot_each, n_pilots, min_frames

def try_read_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        txt = f.read()
    csv_txt = slice_to_csv_from_header(txt)
    rdr = csv.DictReader(io.StringIO(csv_txt))
    return [r for r in rdr]

def find_nonce_row(index_csv: str, sha_hex: str):
    if not sha_hex or not os.path.isfile(index_csv):
        return None
    try:
        rows = try_read_rows(index_csv)
    except Exception:
        return None
    for row in reversed(rows):
        if (row.get("outfile_sha256") or "").lower() == sha_hex.lower():
            return row
    return None

def find_embed_row(embed_csv_path: str, infile_path: str, match_order: List[str]) -> tuple:
    debug = {"searched_csv": embed_csv_path, "strategy": None}
    if not os.path.isfile(embed_csv_path):
        return None, debug
    try:
        rows = try_read_rows(embed_csv_path)
    except Exception:
        return None, debug

    infile_norm = normpath_ci(infile_path)
    infile_base = os.path.basename(infile_path)

    in_sr = in_ch = in_dur = None
    try:
        in_sr, in_ch, in_dur = audio_meta(infile_path)
    except Exception:
        pass

    infile_sha = None
    try:
        infile_sha = sha256_file(infile_path)
    except Exception:
        pass

    for strat in match_order:
        if strat == "sha256" and infile_sha:
            for row in reversed(rows):
                if (row.get("outfile_sha256") or "").lower() == infile_sha.lower():
                    debug["strategy"] = f"sha256:{infile_sha[:12]}..."
                    return row, debug
        if strat == "fullpath":
            for row in reversed(rows):
                outp = (row.get("outfile") or "").strip().strip('"')
                if outp and normpath_ci(outp) == infile_norm:
                    debug["strategy"] = "fullpath"
                    return row, debug
        if strat == "basename_meta" and in_sr and in_ch and in_dur:
            for row in reversed(rows):
                outp = (row.get("outfile") or "").strip().strip('"')
                if not outp or os.path.basename(outp) != infile_base:
                    continue
                try:
                    r_sr = int(row.get("in_sr") or row.get("sr") or 0)
                    r_ch = int(row.get("channels") or 0)
                    r_du = float(row.get("duration_s") or 0.0)
                except Exception:
                    continue
                if r_sr == in_sr and r_ch == in_ch and abs(r_du - in_dur) <= 0.02:
                    debug["strategy"] = "basename_meta"
                    return row, debug
    return None, debug

# --------- Logging ---------
def append_log_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["timestamp","infile","ok","error","det_score_mean","frames_detected",
              "pilot_frames_marked","data_frames_after_pilots","collapsed_frames",
              "key_id","type","nonce_hex8","outfile"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def fail(outfile, log_csv, infile, msg, extras=None, per_frame=None, is_pilot=None,
         data_frames=None, collapsed_frames=None, kid=None, typ=None, pnonce_hex=None):
    result = {"ok": 0, "error": msg}
    if extras: result.update(extras)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if log_csv:
        det_scores = [float(s) for (_, _, s, _, _) in (per_frame or [])]
        append_log_csv(log_csv, {
            "timestamp": datetime.utcnow().isoformat(),
            "infile": infile, "ok": 0, "error": msg,
            "det_score_mean": round(float(np.mean(det_scores)) if det_scores else 0.0, 4),
            "frames_detected": len(per_frame or []),
            "pilot_frames_marked": sum(1 for v in (is_pilot or []) if v),
            "data_frames_after_pilots": len(data_frames or []),
            "collapsed_frames": len(collapsed_frames or []),
            "key_id": kid or "", "type": typ or "", "nonce_hex8": pnonce_hex or "",
            "outfile": outfile
        })
    print(f"Wrote: {outfile}")
    sys.exit(2)

# --------- Detection core ---------
def detect_segment_bits(detector, seg_16k: torch.Tensor) -> Tuple[float, List[int]]:
    seg_b = seg_16k.unsqueeze(0).unsqueeze(0)
    score, msg = detector.detect_watermark(seg_b, AS_SR)
    if isinstance(msg, torch.Tensor):
        msg_bits = [1 if float(v) >= 0.5 else 0 for v in msg.squeeze().tolist()]
    else:
        msg_bits = list(map(int, msg))
    if len(msg_bits) < 16: msg_bits += [0] * (16 - len(msg_bits))
    if len(msg_bits) > 16: msg_bits = msg_bits[:16]
    return float(score), msg_bits

def best_jitter_window(detector, mono_m: torch.Tensor, s_base: int, seg_len_m: int,
                       jitter_m: int, step: int) -> Tuple[float, List[int], int]:
    best = (-1.0, [0]*16, s_base)
    lo = -jitter_m // 2; hi = jitter_m // 2
    if step <= 0: step = max(1, jitter_m // 8 or 1)
    for off in range(lo, hi+1, step):
        s = max(0, min(s_base + off, mono_m.numel() - seg_len_m))
        e = s + seg_len_m
        score, bits = detect_segment_bits(detector, mono_m[s:e].contiguous())
        if score > best[0]:
            best = (score, bits, s)
    return best

def main():
    ap = argparse.ArgumentParser(description="Detector for wm_singlepath (v7)")
    ap.add_argument("--in", dest="infile", required=True, help="Input WAV/FLAC/AIFF/AIF/MP3/OGG")
    base = None  # for outfile naming
    ap.add_argument("--out", dest="outfile", default=None, help="Output JSON (default: Logs/<basename>.wm.json)")
    ap.add_argument("--json", dest="json_out", default=None, help="Alias for --out")

    ap.add_argument("--log-csv", dest="log_csv", default=None, help="Detect log CSV (default: Logs/wm_detect_log.csv)")
    ap.add_argument("--embed-log", dest="embed_log", default=None, help="Embed CSV to read (default: Logs/wm_singlepath_log.csv)")
    ap.add_argument("--pull-embed-params", action="store_true",
                    help="Adopt seg/repeat/sync-every/pilot-bits/fec/interleave_depth from matched row")
    ap.add_argument("--nonce-hex", dest="nonce_hex", default=None, help="Override: 8 hex chars (e.g., 2c7b9e6e)")

    ap.add_argument("--match-order", default="sha256,fullpath,basename_meta",
                    help="Comma list: sha256, fullpath, basename_meta")
    ap.add_argument("--strict-sha", action="store_true", help="Use only sha256 matching")
    ap.add_argument("--trace-match", action="store_true", help="Print which CSV row was matched")

    ap.add_argument("--seg", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--sync-every", type=int, default=32)
    ap.add_argument("--pilot-bits", type=int, default=32)
    ap.add_argument("--energy-gate-db", type=float, default=-20.0)
    ap.add_argument("--min-spacing", type=int, default=2)
    ap.add_argument("--max-hits-per-seg", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--jitter-step-ms", type=int, default=10)
    ap.add_argument("--det-thresh", type=float, default=0.50)
    ap.add_argument("--pilot-hd", type=int, default=2)
    ap.add_argument("--fec", choices=["none","hamming15","bch63"], default="none")
    ap.add_argument("--interleave-depth", type=int, default=8)

    ap.add_argument("--key-id", type=int, default=1)
    ap.add_argument("--key", default="")
    ap.add_argument("--auth-key", default="")
    ap.add_argument("--secret", default="", help="Single secret to derive Kp/Ks/Ka")

    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args = ap.parse_args()

    default_embed_path = os.path.abspath(default_embed_csv())
    args.embed_log = os.path.abspath(args.embed_log or default_embed_path)
    args.log_csv  = os.path.abspath(args.log_csv  or default_detect_csv())
    os.makedirs(os.path.dirname(args.embed_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)

    base = os.path.splitext(os.path.basename(args.infile))[0]
    outfile = args.json_out or args.outfile or log_path(base + ".wm.json")

    # ------ Pull nonce/params from CSV if not given ------
    nonce_csv_path = os.path.abspath(nonce_index_csv())
    match_strategy = None
    matched_row = None
    row_source = None
    infile_sha = None
    try:
        infile_sha = sha256_file(args.infile)
    except Exception:
        infile_sha = None

    if not args.nonce_hex and infile_sha:
        nonce_row = find_nonce_row(nonce_csv_path, infile_sha)
        if nonce_row:
            matched_row = nonce_row
            row_source = nonce_csv_path
            match_strategy = "nonce_index"
            args.nonce_hex = (nonce_row.get("nonce_hex8") or "").strip()
            if args.pull_embed_params:
                try:
                    if nonce_row.get("seg"): args.seg = float(nonce_row["seg"])
                    if nonce_row.get("repeat"): args.repeat = int(nonce_row["repeat"])
                    if nonce_row.get("sync_every"): args.sync_every = int(nonce_row["sync_every"])
                    if nonce_row.get("pilot_bits"): args.pilot_bits = int(nonce_row["pilot_bits"])
                    if nonce_row.get("fec"): args.fec = (nonce_row["fec"] or "none").strip().lower()
                    if nonce_row.get("interleave_depth"): args.interleave_depth = int(nonce_row["interleave_depth"])
                except Exception as e:
                    fail(outfile, args.log_csv, args.infile, f"Failed to pull params from nonce index: {e}",
                         extras={"debug": {"row_src": nonce_csv_path, "row": nonce_row}})

    row_dbg = None
    if not args.nonce_hex:
        match_order = ["sha256"] if args.strict_sha else [s.strip() for s in (args.match_order or "").split(",") if s.strip()]
        row, dbg = find_embed_row(args.embed_log, args.infile, match_order)
        row_dbg = dbg
        if row:
            matched_row = row
            row_source = args.embed_log
            match_strategy = dbg.get("strategy")
        if not row and args.embed_log != default_embed_path:
            row, dbg_fallback = find_embed_row(default_embed_path, args.infile, match_order)
            if row:
                matched_row = row
                row_source = default_embed_path
                match_strategy = (dbg_fallback.get("strategy") or "") + " (fallback)"
                dbg = dbg_fallback
                args.embed_log = default_embed_path
        if not matched_row:
            fail(outfile, args.log_csv, args.infile,
                 f"No matching row in embed CSV ({args.embed_log}). Pass --nonce-hex or ensure hash is logged.",
                 extras={"debug": {"embed_csv": args.embed_log, "nonce_index": nonce_csv_path, "match": row_dbg}})
        args.nonce_hex = (matched_row.get("nonce_hex8") or "").strip()
        if args.pull_embed_params:
            try:
                args.seg = float(matched_row["seg"]); args.repeat = int(matched_row["repeat"])
                args.sync_every = int(matched_row["sync_every"]); args.pilot_bits = int(matched_row["pilot_bits"])
                args.fec = (matched_row["fec"] or "none").strip().lower(); args.interleave_depth = int(matched_row["interleave_depth"])
            except Exception as e:
                fail(outfile, args.log_csv, args.infile, f"Failed to pull params from CSV: {e}",
                     extras={"debug": {"row_src": row_source, "row": matched_row}})

    if args.trace_match and matched_row:
        print(f"[match] source={row_source}\n"
              f"[match] strategy={match_strategy}\n"
              f"[match] outfile={matched_row.get('outfile')}\n"
              f"[match] sha256={matched_row.get('outfile_sha256')}\n"
              f"[match] nonce={matched_row.get('nonce_hex8')}")

    if not args.nonce_hex or len(args.nonce_hex) != 8:
        fail(outfile, args.log_csv, args.infile,
             "Missing/invalid nonce. Provide --nonce-hex or ensure embed CSV has nonce_hex8.",
             extras={"debug": {"csv": args.embed_log, "strategy": match_strategy}})

    try:
        salt = bytes.fromhex(args.nonce_hex)[-4:]
    except Exception:
        fail(outfile, args.log_csv, args.infile, "Could not parse --nonce-hex (must be 8 hex digits).")

    # ------ Read + resample ------
    try:
        audio_np, sr_in = read_audio(args.infile)
    except Exception as e:
        fail(outfile, args.log_csv, args.infile, f"Failed to read audio: {e}")

    mono_m = resample_mono(audio_np, sr_in)
    seg_sec = max(MIN_SEG_SEC, float(args.seg))
    seg_len_m = int(seg_sec * AS_SR)
    n_segments = max(1, mono_m.numel() // seg_len_m)
    jitter_m = int(max(0.0, float(args.jitter)) * AS_SR)
    step = max(1, int(args.jitter_step_ms * AS_SR / 1000.0))

    # ------ Keys (salted with nonce) ------
    if args.secret:
        base_seed = hashlib.sha256(args.secret.encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
        auth_key = hkdf_sha256(base_seed, salt=salt, info=b"wm.auth", length=32)
    else:
        base_seed = hashlib.sha256((args.key or "").encode("utf-8") + bytes([args.key_id & 0xFF])).digest()
        auth_key = (args.auth_key or "").encode("utf-8")
    Kp = hkdf_sha256(base_seed, salt=salt, info=b"wm.place", length=32)
    Ks = hkdf_sha256(base_seed, salt=salt, info=b"wm.sync", length=32)

    # ------ Candidate pool + scoring ------
    pool = energy_gated_indices(mono_m, seg_len_m, gate_db=args.energy_gate_db)
    scorer = SpectralScorer(seg_len_m, device=args.device)
    scores = scorer.scores(mono_m)
    if len(pool) < n_segments: pool = list(range(n_segments))

    order = schedule_positions(pool=pool, total_needed=len(pool), min_spacing=args.min_spacing,
                               seed=Kp, scores=scores, max_hits_per_seg=args.max_hits_per_seg, n_segments=n_segments)
    rank = {seg: i for i, seg in enumerate(order)}

    # ------ Detector ------
    try:
        from audioseal import AudioSeal
        detector = AudioSeal.load_detector(DET_MODEL_NAME)
    except Exception as e:
        fail(outfile, args.log_csv, args.infile, f"Failed to load AudioSeal detector: {e}")

    all_frames = []
    for seg_idx in order:
        base = seg_idx * seg_len_m
        best_score, bits, s_used = best_jitter_window(detector, mono_m, base, seg_len_m, jitter_m, step)
        all_frames.append((rank[seg_idx], seg_idx, best_score, bits, s_used))

    frames_total_est, frames_pilot_each_est, n_pilots_est, min_frames_required = estimate_frame_requirements(
        matched_row or {}, mono_m.numel(), seg_len_m, args
    )

    if min_frames_required > len(all_frames):
        min_frames_required = max(1, len(all_frames))

    thresholds = [
        args.det_thresh,
        max(0.0, args.det_thresh * 0.75),
        max(0.0, args.det_thresh * 0.5),
        0.25,
        0.0,
    ]
    seen = []
    adaptive_thresholds = []
    for t in thresholds:
        if t not in seen:
            adaptive_thresholds.append(t); seen.append(t)

    per_frame = []
    used_threshold = None
    for thr in adaptive_thresholds:
        cand = [entry for entry in all_frames if entry[2] >= thr]
        if len(cand) >= min_frames_required:
            per_frame = cand
            used_threshold = thr
            break
    if not per_frame:
        sorted_all = sorted(all_frames, key=lambda t: t[2], reverse=True)
        take_n = max(1, min(min_frames_required, len(sorted_all)))
        per_frame = sorted_all[:take_n]
        used_threshold = None

    per_frame.sort(key=lambda t: t[0])
    frame_debug = {
        "frames_total_est": frames_total_est,
        "frames_selected": len(per_frame),
        "threshold_used": used_threshold if used_threshold is not None else "adaptive",
        "min_required": min_frames_required,
        "pilot_est": n_pilots_est,
    }
    per_frame.sort(key=lambda t: t[0])
    messages = [bits for (_, _, _, bits, _) in per_frame]

    pilot_bits = derive_pilot_bits(Ks, max(0, int(args.pilot_bits)))
    pilot_frames = bits_to_frames(pilot_bits, width=16); pf = len(pilot_frames)

    def looks_like_pilot(start: int) -> bool:
        if start + pf > len(messages): return False
        for i in range(pf):
            if hamming(messages[start + i], pilot_frames[i]) > args.pilot_hd: return False
        return True

    is_pilot = [False] * len(messages); i = 0; count_pilots = 0
    while i < len(messages):
        if looks_like_pilot(i):
            for k in range(pf): is_pilot[i+k] = True
            count_pilots += 1; i += pf
        else:
            i += 1
    if count_pilots == 0:
        fail(outfile, args.log_csv, args.infile,
             "0 pilot frames matched. Nonce or params likely wrong.",
             extras={"debug": frame_debug},
             per_frame=per_frame, is_pilot=is_pilot)

    data_frames = [m for m, p in zip(messages, is_pilot) if not p]

    rep = max(1, int(args.repeat))
    collapsed_frames: List[List[int]] = []
    for j in range(0, len(data_frames) // rep * rep, rep):
        grp = data_frames[j:j+rep]
        collapsed_frames.append(majority_bits(grp))

    bitstream = frames_to_bits(collapsed_frames)
    deint_bits = block_deinterleave(bitstream, args.interleave_depth)
    raw_bits = remove_fec(deint_bits, args.fec)
    payload_bytes = pack_bits_to_bytes(raw_bits)

    try:
        header, body, kid, typ, flags, pnonce, ver, tag, crc = parse_payload(payload_bytes)
    except Exception as e:
        fail(outfile, args.log_csv, args.infile, f"Failed to parse payload: {e}",
             per_frame=per_frame, is_pilot=is_pilot, data_frames=data_frames, collapsed_frames=collapsed_frames)

    authenticated = 0
    if flags & 0x01:
        if args.secret:
            re_auth_key = hkdf_sha256(hashlib.sha256(args.secret.encode("utf-8") + bytes([args.key_id & 0xFF])).digest(),
                                      salt=pnonce.to_bytes(4, "big"), info=b"wm.auth", length=32)
        else:
            re_auth_key = (args.auth_key or "").encode("utf-8")
        head_wo_tag = bytes([ver, kid, typ, flags]) + pnonce.to_bytes(4, "big") + header["len"].to_bytes(4, "big") + body
        calc = hmac.new(re_auth_key, head_wo_tag, hashlib.sha256).digest()[:16]
        if calc != tag:
            fail(outfile, args.log_csv, args.infile, "HMAC tag mismatch",
                 per_frame=per_frame, is_pilot=is_pilot, data_frames=data_frames,
                 collapsed_frames=collapsed_frames, kid=kid, typ=typ, pnonce_hex=f"{pnonce:08x}")
        authenticated = 1

    manifest = None
    if typ == 2:
        try:
            manifest = json.loads(zlib.decompress(body).decode("utf-8"))
        except Exception:
            manifest = None

    det_scores = [float(s) for (_, _, s, _, _) in per_frame]
    pilot_hits = sum(1 for v in is_pilot if v)
    result = {
        "ok": 1, "type": typ, "key_id": kid, "nonce_hex8": f"{pnonce:08x}",
        "authenticated": authenticated, "payload_len": len(body),
        "header": header,
        "stats": {
            "segments_total": n_segments,
            "frames_detected": len(messages),
            "pilot_frames_marked": pilot_hits,
            "data_frames_after_pilots": len(data_frames),
            "collapsed_frames": len(collapsed_frames),
            "det_score_mean": round(float(np.mean(det_scores)) if det_scores else 0.0, 4),
        },
        "manifest": manifest,
        "notes": "Placement-order scan + jitter search; pilots stripped via nonce-keyed match.",
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    append_log_csv(default_detect_csv() if args.log_csv is None else args.log_csv, {
        "timestamp": datetime.utcnow().isoformat(),
        "infile": args.infile, "ok": 1, "error": "",
        "det_score_mean": result["stats"]["det_score_mean"],
        "frames_detected": result["stats"]["frames_detected"],
        "pilot_frames_marked": result["stats"]["pilot_frames_marked"],
        "data_frames_after_pilots": result["stats"]["data_frames_after_pilots"],
        "collapsed_frames": result["stats"]["collapsed_frames"],
        "key_id": result["key_id"], "type": result["type"],
        "nonce_hex8": result["nonce_hex8"],
        "outfile": outfile
    })

    print(f"Wrote: {outfile}")
    sys.exit(0)

if __name__ == "__main__":
    main()
