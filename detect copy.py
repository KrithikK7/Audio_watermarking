#!/usr/bin/env python3
"""
detect.py — Detector for wm_singlepath (v7)

Features:
  • --json, --log-csv (wrapper-friendly)
  • --embed-log <csv or folder>  → auto-pick nonce for THIS file
  • --pull-embed-params          → adopt seg/repeat/sync-every/pilot-bits/fec/interleave_depth from CSV
  • Auto-discovery if --embed-log not provided (search near the input file)
  • --match-order / --strict-sha → control how the CSV row is matched (sha256 only if you want)
  • Optional --trace-match to print which row was used

Exit codes: 0 = success, 2 = failure
"""
import argparse, json, os, sys, zlib, hashlib, hmac, csv, io, glob, math
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import soundfile as sf
try:
    from torchaudio.transforms import Resample
except Exception:
    Resample = None

# --- Import helpers from your embedder (embed.py must be alongside this file) ---
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
    w = len(frames[0])
    out = []
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

# --------- Embed log discovery & matching ---------
def discover_embed_csvs(hint: Optional[str], infile_path: str) -> List[str]:
    """Return candidate CSV paths (most recent first)."""
    candidates = []
    if hint:
        if os.path.isdir(hint):
            patterns = ["*wm*singlepath*log*.csv", "wm_singlepath_log.csv", "*wm*log*.csv", "*.csv"]
            for pat in patterns:
                candidates += glob.glob(os.path.join(hint, pat))
        elif os.path.isfile(hint):
            candidates.append(hint)
    else:
        base_dir = os.path.dirname(infile_path)
        dirs = [base_dir, os.path.dirname(base_dir)]
        patterns = ["*wm*singlepath*log*.csv", "wm_singlepath_log.csv", "*wm*log*.csv"]
        for d in dirs:
            for pat in patterns:
                candidates += glob.glob(os.path.join(d, pat))
    uniq = []
    seen = set()
    for p in candidates:
        q = os.path.abspath(p)
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    uniq.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return uniq

def try_read_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        txt = f.read()
    csv_txt = slice_to_csv_from_header(txt)
    rdr = csv.DictReader(io.StringIO(csv_txt))
    return [r for r in rdr]

def find_embed_row(candidates: List[str], infile_path: str, match_order: List[str]) -> tuple:
    """Return (row, source_csv, debug) or (None, last_csv, debug)"""
    debug = {"searched_csvs": candidates, "strategy": None}
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

    last_csv = None
    for csv_path in candidates:
        last_csv = csv_path
        try:
            rows = try_read_rows(csv_path)
        except Exception:
            continue

        for strat in match_order:
            if strat == "sha256" and infile_sha:
                for row in reversed(rows):
                    if (row.get("outfile_sha256") or "").lower() == infile_sha.lower():
                        debug["strategy"] = f"sha256:{infile_sha[:12]}..."
                        return row, csv_path, debug

            if strat == "fullpath":
                for row in reversed(rows):
                    outp = (row.get("outfile") or "").strip().strip('"')
                    if outp and normpath_ci(outp) == infile_norm:
                        debug["strategy"] = "fullpath"
                        return row, csv_path, debug

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
                        return row, csv_path, debug

    return None, last_csv, debug

# --------- Logging ---------
def append_log_csv(csv_path: str, row: dict):
    fields = ["timestamp","infile","ok","error",
              "det_score_mean","frames_detected","pilot_frames_marked",
              "data_frames_after_pilots","collapsed_frames",
              "key_id","type","nonce_hex8","outfile"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def fail(outfile, log_csv, infile, msg, extras=None, per_frame=None, is_pilot=None, data_frames=None, collapsed_frames=None, kid=None, typ=None, pnonce_hex=None):
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
            "key_id": kid or "",
            "type": typ or "",
            "nonce_hex8": pnonce_hex or "",
            "outfile": outfile
        })
    print(f"Wrote: {outfile}")
    sys.exit(2)

# --------- Detection core ---------
def detect_segment_bits(detector, seg_16k: torch.Tensor) -> Tuple[float, List[int]]:
    seg_b = seg_16k.unsqueeze(0).unsqueeze(0)  # [1,1,T]
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
    lo = -jitter_m // 2
    hi =  jitter_m // 2
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
    ap.add_argument("--out", dest="outfile", default=None, help="Output JSON (default: <in>.wm.json)")
    ap.add_argument("--json", dest="json_out", default=None, help="Alias for --out")
    ap.add_argument("--log-csv", dest="log_csv", default=None, help="Append one-line CSV log")

    # Embed CSV/nonce handling
    ap.add_argument("--embed-log", dest="embed_log", default=None, help="Embed CSV file OR folder to search")
    ap.add_argument("--pull-embed-params", action="store_true",
                    help="Adopt seg/repeat/sync-every/pilot-bits/fec/interleave_depth from the matched CSV row")
    ap.add_argument("--nonce-hex", dest="nonce_hex", default=None, help="Override: 8 hex chars from embed log (e.g., 2c7b9e6e)")

    # Matching control
    ap.add_argument("--match-order", default="sha256,fullpath,basename_meta",
                    help="Comma list of strategies. Options: sha256, fullpath, basename_meta")
    ap.add_argument("--strict-sha", action="store_true", help="Shortcut for --match-order sha256 (no filename/path fallback)")
    ap.add_argument("--trace-match", action="store_true", help="Print which CSV row was matched")

    # Embed-time params (can be auto-pulled)
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

    # Keys
    ap.add_argument("--key-id", type=int, default=1)
    ap.add_argument("--key", default="")
    ap.add_argument("--auth-key", default="")
    ap.add_argument("--secret", default="", help="Single secret to derive Kp/Ks/Ka (overrides --key/--auth-key)")

    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args = ap.parse_args()

    outfile = args.json_out or args.outfile or (os.path.splitext(args.infile)[0] + ".wm.json")

    # ------ Pull nonce (and optionally params) from CSV ------
    row = None
    used_csv = None
    match_strategy = None

    if not args.nonce_hex:
        candidates = discover_embed_csvs(args.embed_log, args.infile)
        if args.strict_sha:
            match_order = ["sha256"]
        else:
            match_order = [s.strip() for s in (args.match_order or "").split(",") if s.strip()]
        if not candidates:
            fail(outfile, args.log_csv, args.infile,
                 "No embed CSV found near file. Pass --nonce-hex or --embed-log <csv/folder>.",
                 extras={"debug": {"searched_csvs": []}})

        row, used_csv, dbg = find_embed_row(candidates, args.infile, match_order)
        match_strategy = dbg.get("strategy")
        if not row:
            fail(outfile, args.log_csv, args.infile,
                 "Could not find matching row in embed CSV with the selected match-order.",
                 extras={"debug": {"searched_csvs": candidates, "strategy": match_order}})

        args.nonce_hex = (row.get("nonce_hex8") or "").strip()
        if args.pull_embed_params:
            try:
                args.seg = float(row["seg"])
                args.repeat = int(row["repeat"])
                args.sync_every = int(row["sync_every"])
                args.pilot_bits = int(row["pilot_bits"])
                args.fec = (row["fec"] or "none").strip().lower()
                args.interleave_depth = int(row["interleave_depth"])
            except Exception as e:
                fail(outfile, args.log_csv, args.infile,
                     f"Failed to pull embed params from CSV: {e}",
                     extras={"debug": {"row_src": used_csv, "row": row}})

        if args.trace_match and row:
            print(f"[match] csv={used_csv}\n"
                  f"[match] strategy={match_strategy}\n"
                  f"[match] outfile={row.get('outfile')}\n"
                  f"[match] sha256={row.get('outfile_sha256')}\n"
                  f"[match] nonce={row.get('nonce_hex8')}")

    if not args.nonce_hex or len(args.nonce_hex) != 8:
        fail(outfile, args.log_csv, args.infile,
             "Missing/invalid nonce. Provide --nonce-hex or ensure embed CSV has nonce_hex8.",
             extras={"debug": {"csv": used_csv, "strategy": match_strategy}})

    try:
        salt = bytes.fromhex(args.nonce_hex)[-4:]  # 4 bytes salt used by embedder
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

    # ------ Keys with correct salt ------
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
    if len(pool) < n_segments:
        pool = list(range(n_segments))

    order = schedule_positions(
        pool=pool, total_needed=len(pool), min_spacing=args.min_spacing,
        seed=Kp, scores=scores, max_hits_per_seg=args.max_hits_per_seg, n_segments=n_segments
    )
    rank = {seg: i for i, seg in enumerate(order)}

    # ------ Detector ------
    try:
        from audioseal import AudioSeal
        detector = AudioSeal.load_detector(DET_MODEL_NAME)
    except Exception as e:
        fail(outfile, args.log_csv, args.infile, f"Failed to load AudioSeal detector: {e}")

    per_frame = []
    for seg_idx in order:
        base = seg_idx * seg_len_m
        best_score, bits, s_used = best_jitter_window(detector, mono_m, base, seg_len_m, jitter_m, step)
        if best_score >= args.det_thresh:
            per_frame.append((rank[seg_idx], seg_idx, best_score, bits, s_used))

    if not per_frame:
        extras = {"debug": {"csv": used_csv, "strategy": match_strategy}}
        if row:
            extras["embed_params"] = {
                "seg": row.get("seg"), "repeat": row.get("repeat"),
                "sync_every": row.get("sync_every"), "pilot_bits": row.get("pilot_bits"),
                "fec": row.get("fec"), "interleave_depth": row.get("interleave_depth"),
            }
        fail(outfile, args.log_csv, args.infile,
             "No segments passed threshold. Likely param mismatch or wrong secret.",
             extras=extras)

    per_frame.sort(key=lambda t: t[0])
    messages = [bits for (_, _, _, bits, _) in per_frame]

    # ------ Pilots (Ks) ------
    pilot_bits = derive_pilot_bits(Ks, max(0, int(args.pilot_bits)))
    pilot_frames = bits_to_frames(pilot_bits, width=16)
    pf = len(pilot_frames)

    def looks_like_pilot(start: int) -> bool:
        if start + pf > len(messages): return False
        for i in range(pf):
            if hamming(messages[start + i], pilot_frames[i]) > args.pilot_hd:
                return False
        return True

    is_pilot = [False] * len(messages)
    i = 0
    count_pilots = 0
    while i < len(messages):
        if looks_like_pilot(i):
            for k in range(pf): is_pilot[i+k] = True
            count_pilots += 1
            i += pf
        else:
            i += 1

    if count_pilots == 0:
        fail(outfile, args.log_csv, args.infile,
             "0 pilot frames matched. Nonce or params likely wrong.",
             per_frame=per_frame, is_pilot=is_pilot)

    data_frames = [m for m, p in zip(messages, is_pilot) if not p]

    # ------ De-repetition ------
    rep = max(1, int(args.repeat))
    collapsed_frames: List[List[int]] = []
    for j in range(0, len(data_frames) // rep * rep, rep):
        grp = data_frames[j:j+rep]
        collapsed_frames.append(majority_bits(grp))

    # ------ Bits -> deinterleave -> FEC- -> bytes ------
    bitstream = frames_to_bits(collapsed_frames)
    deint_bits = block_deinterleave(bitstream, args.interleave_depth)
    raw_bits = remove_fec(deint_bits, args.fec)
    payload_bytes = pack_bits_to_bytes(raw_bits)

    # ------ Parse payload ------
    try:
        header, body, kid, typ, flags, pnonce, ver, tag, crc = parse_payload(payload_bytes)
    except Exception as e:
        fail(outfile, args.log_csv, args.infile, f"Failed to parse payload: {e}",
             per_frame=per_frame, is_pilot=is_pilot, data_frames=data_frames, collapsed_frames=collapsed_frames)

    # ------ Verify HMAC if present ------
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

    # ------ Inflate manifest for TYPE_FULL ------
    manifest = None
    if typ == 2:
        try:
            manifest = json.loads(zlib.decompress(body).decode("utf-8"))
        except Exception:
            manifest = None

    det_scores = [float(s) for (_, _, s, _, _) in per_frame]
    pilot_hits = sum(1 for v in is_pilot if v)
    result = {
        "ok": 1,
        "type": typ,
        "key_id": kid,
        "nonce_hex8": f"{pnonce:08x}",
        "authenticated": authenticated,
        "payload_len": len(body),
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

    if args.log_csv:
        append_log_csv(args.log_csv, {
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
