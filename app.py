import gradio as gr
import json
import os
import csv
import re
import torch
import math
import tempfile
from google import genai
from google.genai import types as genai_types
from transformers import pipeline as hf_pipeline

# ==========================================
# Globals
# ==========================================
_nli_model          = None
_current_hypotheses = []
_current_weights    = []
_scored_cache       = {}
_original_data      = []

# ==========================================
# Demo Data
# ==========================================
DEMO_CANDIDATES = json.dumps([
    {
        "candidate_id": "DEMO_001",
        "profile": {"current_title": "Senior Backend Engineer", "years_of_experience": 6,
                    "summary": "Experienced in distributed microservices, Kubernetes, and Python."},
        "career_history": [
            {"title": "Backend Engineer", "company": "TechCorp",
             "description": "Designed and maintained distributed microservices on AWS. Led migration to Kubernetes."},
            {"title": "Junior Developer", "company": "StartupXYZ",
             "description": "Developed REST APIs and contributed to CI/CD pipelines."}
        ],
        "redrob_signals": {"recruiter_response_rate": 0.82, "github_activity_score": 78}
    },
    {
        "candidate_id": "DEMO_002",
        "profile": {"current_title": "Full Stack Developer", "years_of_experience": 3,
                    "summary": "React, Node.js, some Docker exposure."},
        "career_history": [
            {"title": "Full Stack Dev", "company": "AgencyA",
             "description": "Built web apps in React. Limited backend architecture experience."}
        ],
        "redrob_signals": {"recruiter_response_rate": 0.45, "github_activity_score": 31}
    },
    {
        "candidate_id": "DEMO_003",
        "profile": {"current_title": "Platform Engineer", "years_of_experience": 8,
                    "summary": "Deep infrastructure background, Terraform, Kubernetes, AWS cost optimisation."},
        "career_history": [
            {"title": "Platform Engineer", "company": "FinTechCo",
             "description": "Led Kubernetes cluster upgrades, wrote Terraform modules, reduced infra cost by 40%."},
            {"title": "DevOps Engineer", "company": "MegaCorp",
             "description": "Maintained CI/CD pipelines and automated deployments for 200+ microservices."}
        ],
        "redrob_signals": {"recruiter_response_rate": 0.91, "github_activity_score": 92}
    },
    {
        "candidate_id": "DEMO_004",
        "profile": {"current_title": "Python Developer", "years_of_experience": 2,
                    "summary": "Python scripting and data pipelines. Learning Docker."},
        "career_history": [
            {"title": "Python Developer", "company": "DataStartup",
             "description": "Wrote ETL scripts. No distributed systems exposure yet."}
        ],
        "redrob_signals": {"recruiter_response_rate": 0.12, "github_activity_score": 15}
    },
    {
        "candidate_id": "DEMO_005",
        "profile": {"current_title": "Solutions Architect", "years_of_experience": 10,
                    "summary": "AWS certified architect. Designed multi-region microservices and event-driven systems."},
        "career_history": [
            {"title": "Solutions Architect", "company": "CloudVendor",
             "description": "Architected distributed event-driven microservices for Fortune 500 clients. AWS, Kafka, Kubernetes."},
            {"title": "Senior Engineer", "company": "HealthTech",
             "description": "Led backend teams and code review processes. Strong mentoring record."}
        ],
        "redrob_signals": {"recruiter_response_rate": 0.67, "github_activity_score": 55}
    }
])

DEMO_JD = """Senior Backend Engineer — Distributed Systems

We are looking for a Senior Backend Engineer with deep expertise in distributed microservices
and cloud-native infrastructure. The ideal candidate will have:

- 5+ years of hands-on experience with distributed microservices architectures
- Strong proficiency with Kubernetes and container orchestration
- Experience with AWS or GCP cloud platforms
- Proven track record of leading backend teams and code reviews
- Familiarity with CI/CD pipelines and DevOps practices
- Ability to optimise infrastructure costs at scale
"""

MAX_HYPOTHESES = 10

# ==========================================
# NLI Model
# ==========================================
def get_nli_model():
    global _nli_model
    if _nli_model is None:
        device = 0 if torch.cuda.is_available() else -1
        _nli_model = hf_pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=device
        )
    return _nli_model

# ==========================================
# Helpers
# ==========================================
def clamp_signals(signals: dict) -> dict:
    """Return a copy of signals with all values clamped to valid ranges."""
    c = dict(signals)
    if "recruiter_response_rate" in c:
        c["recruiter_response_rate"] = max(0.0, min(1.0, float(c["recruiter_response_rate"])))
    if "github_activity_score" in c:
        c["github_activity_score"]   = max(0.0, min(100.0, float(c["github_activity_score"])))
    return c

def detect_red_flags(signals, nli_scores):
    flags = []
    rr = max(0.0, min(1.0, signals.get("recruiter_response_rate", 1.0)))
    gh = max(0.0, signals.get("github_activity_score", 100.0))
    if rr < 0.20:
        flags.append("LOW ENGAGEMENT")
    if gh < 20:
        flags.append("LOW GITHUB")
    if len([s for s in nli_scores if s < 0.15]) >= 2:
        flags.append("SKILL GAP")
    return flags

def _score_bucket(score):
    if score >= 0.70: return "Strong"
    if score >= 0.40: return "Moderate"
    if score >= 0.15: return "Weak"
    return "Unsupported"

def build_exec_summary(candidate, as_html=False):
    """
    Builds the reasoning string.
    If as_html=True, injects spans and breaks for the UI.
    If as_html=False, keeps it strictly plain-text for CSV exports.
    """
    flags      = candidate.get("red_flags", [])
    hyp_scores = candidate.get("nli_scores", [])
    hyp_labels = candidate.get("nli_labels", [])
    weights    = candidate.get("nli_weights", [1.0] * len(hyp_scores))
    signals    = candidate.get("signals", {})
    weighted   = candidate.get("nli_weighted_avg", 0.0)

    # Flag Formatting
    raw_flag = ("⚠ " + ", ".join(flags)) if flags else "✅ Clear"
    if as_html:
        flag_str = f"<span style='color:#da3633'><b>[{raw_flag}]</b></span>" if flags else f"<span style='color:#3fb950'><b>[{raw_flag}]</b></span>"
    else:
        flag_str = f"[{raw_flag}]"

    # NLI Formatting
    if hyp_scores:
        pairs = list(zip(hyp_labels, hyp_scores, weights))
        best  = max(pairs, key=lambda p: p[1])
        worst = min(pairs, key=lambda p: p[1])

        def _fmt(p):
            label, score, w = p
            tag    = f" [w={w:g}]" if w != 1.0 else ""
            bucket = _score_bucket(score)

            if as_html:
                if bucket == "Strong": b_color = "#3fb950"
                elif bucket == "Moderate": b_color = "#e3b341"
                elif bucket == "Weak" or bucket == "Unsupported": b_color = "#da3633"
                else: b_color = "#8b949e"

                return f'"{label}" → <b style="color:{b_color}">{score:.2f} ({bucket})</b>{tag}'
            return f'"{label}" → {score:.2f} ({bucket}){tag}'

        strongest_str = _fmt(best)
        weakest_str   = _fmt(worst) if worst[0] != best[0] else "—"
    else:
        strongest_str = weakest_str = "n/a"

    # Behavioral Formatting
    rr = round(max(0.0, min(1.0, signals.get("recruiter_response_rate", 0))) * 100)
    gh = int(max(0.0, min(100.0, signals.get("github_activity_score", 0))))

    if as_html:
        return (
            f"{flag_str}<br>"
            f"<b style='color:#58a6ff'>Strongest:</b> {strongest_str}<br>"
            f"<b style='color:#bc8cff'>Weakest:</b> {weakest_str}<br>"
            f"<span style='color:#8b949e'><b>NLI avg:</b> {weighted:.2f} | "
            f"<b>Behavioral:</b> {rr}% response, {gh}/100 GitHub</span>"
        )
    else:
        return (
            f"{flag_str} | "
            f"Strongest: {strongest_str} | "
            f"Weakest: {weakest_str} | "
            f"NLI avg: {weighted:.2f} | "
            f"Behavioral: {rr}% response, {gh}/100 GitHub"
        )

# ==========================================
# Score Spread Helpers
# ==========================================
def minmax_normalize(values):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]

def percentile_ranks(values):
    n = len(values)
    if n == 0:
        return []
    sv = sorted(values)
    return [100.0 * (sum(1 for x in sv if x < v) + 0.5) / n for v in values]

def apply_score_stretch(raw_scores, power=1.5):
    if not raw_scores:
        return []
    normed  = minmax_normalize(raw_scores)
    powered = [s ** power for s in normed]
    return minmax_normalize(powered)

# ==========================================
# Partial-JSON Recovery
# ==========================================
def recover_strings_from_partial_json(text: str) -> list:
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '"':
            i += 1
            continue
        i += 1
        buf = []
        complete = False
        while i < n:
            ch = text[i]
            if ch == '\\' and i + 1 < n:
                nxt = text[i + 1]
                if nxt == '"':  buf.append('"')
                elif nxt == '\\': buf.append('\\')
                elif nxt == 'n':  buf.append('\n')
                elif nxt == 't':  buf.append('\t')
                else:             buf.append(nxt)
                i += 2
            elif ch == '"':
                complete = True
                i += 1
                break
            else:
                buf.append(ch)
                i += 1
        if complete:
            s = ''.join(buf).strip()
            if s:
                results.append(s)
    return results

# ==========================================
# Live Progress HTML Renderer
# ==========================================
def render_progress_html(current, total, label="Scored", bar_width=40, done=False, error=False):
    """
    Builds a centered, monospace ASCII-style progress block as an HTML string,
    meant to sit in the results area above the table while scoring runs.
    """
    if total <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, current / total))

    filled = int(round(pct * bar_width))
    filled = max(0, min(bar_width, filled))
    bar = "█" * filled + "░" * (bar_width - filled)

    if error:
        color = "#da3633"
        title = "⚠ Scoring stopped"
    elif done:
        color = "#3fb950"
        title = "✅ Scoring complete"
    else:
        color = "#58a6ff"
        title = "⏳ Scoring in progress…"

    count_str = f"{current:,} / {total:,}" if total else f"{current:,}"

    return f"""
<div style="display:flex;justify-content:center;align-items:center;padding:18px 8px">
  <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
              padding:20px 28px;text-align:center;min-width:340px">
    <div style="color:{color};font-weight:700;font-size:.85rem;margin-bottom:10px;
                letter-spacing:.04em;text-transform:uppercase">{title}</div>
    <pre style="margin:0;font-family:'JetBrains Mono',monospace;font-size:.95rem;
                color:#c9d1d9;line-height:1.6">{label} {count_str}

[<span style="color:{color}">{bar}</span>]  {pct*100:5.1f}%</pre>
  </div>
</div>
"""

# ==========================================
# STEP 1: Extract Hypotheses
# ==========================================
def extract_hypotheses(api_key, jd_file, num_hypotheses, demo_mode, edit_instruction=""):
    global _current_hypotheses, _current_weights
    logs = []
    def log(m): logs.append(str(m))
    def get_logs(): return "\n".join(logs)

    if demo_mode:
        jd_text = DEMO_JD
        log("🎬 Demo mode: using built-in JD.")
    else:
        if not jd_file:
            log("❌ No JD file provided.")
            return get_logs(), []
        try:
            fp = jd_file if isinstance(jd_file, str) else jd_file.name
            with open(fp, "r", encoding="utf-8") as f:
                jd_text = f.read()
            log(f"📄 JD loaded ({len(jd_text)} chars).")
        except Exception as e:
            log(f"❌ Could not read JD file: {e}")
            return get_logs(), []

    if not api_key:
        log("❌ Please provide a Gemini API key.")
        return get_logs(), []

    try:
        n = max(1, min(MAX_HYPOTHESES, int(num_hypotheses)))
    except Exception:
        n = 5

    extra  = f"\n\nAdditional instruction: {edit_instruction}" if edit_instruction.strip() else ""
    prompt = (
        f"You are an expert technical recruiter. Read the following job description.\n"
        f"Extract exactly {n} logical, verifiable statements (hypotheses) that a candidate "
        f"MUST prove through their resume to be a strong fit for this role.\n\n"
        f"Rules:\n"
        f"- Write complete sentences about CAPABILITY, not just skill keywords.\n"
        f"- Each statement must be independently testable from a resume.\n"
        f"- Example: \"The candidate has designed distributed microservices at scale.\"\n\n"
        f"Job Description:\n{jd_text}{extra}\n\n"
        f"Respond with ONLY a raw JSON array of exactly {n} strings. "
        f"No markdown fences, no backticks, no explanation."
    )

    raw = ""
    try:
        log("🤖 Calling Gemini API…")
        client = genai.Client(api_key=api_key)
        cfg    = genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=cfg,
        )
        raw = (resp.text or "").strip()
        log(f"📨 Raw response received ({len(raw)} chars).")
    except Exception as e:
        log(f"❌ Gemini API call failed: {e}")
        return get_logs(), []

    if not raw:
        log("❌ Gemini returned an empty response. Try regenerating.")
        return get_logs(), []

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s, e    = cleaned.find("["), cleaned.rfind("]")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s : e + 1]

    hyps = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            hyps = [str(h).strip() for h in parsed if str(h).strip()]
    except json.JSONDecodeError as ex:
        log(f"⚠️  Standard JSON parse failed ({ex}); attempting partial recovery…")

    if not hyps:
        recovered = recover_strings_from_partial_json(raw)
        if recovered:
            hyps = recovered
            log(f"🔧 Partial recovery extracted {len(hyps)} string(s) from truncated response.")
        else:
            log(f"❌ Could not extract hypotheses.\nRaw output (first 500 chars):\n{raw[:500]}")
            return get_logs(), []

    if not hyps:
        log(f"❌ Empty hypothesis list after cleaning.\nRaw: {raw[:300]}")
        return get_logs(), []

    hyps = hyps[:MAX_HYPOTHESES]
    if len(hyps) != n:
        log(f"⚠️  Requested {n}, got {len(hyps)} — using all returned.")

    _current_hypotheses = hyps
    _current_weights    = [1.0] * len(hyps)

    log(f"✅ {len(hyps)} hypotheses extracted:")
    for i, h in enumerate(hyps, 1):
        log(f"   {i}. {h}")

    return get_logs(), hyps


# ==========================================
# STEP 2: Score Candidates (generator — yields live progress)
# ==========================================
def run_scoring(candidates_file, demo_mode, hyp_weight_vals):
    """
    Generator version: yields (logs, csv_path, table, radar_json, progress_html)
    repeatedly so the UI can show a live-updating centered progress block
    while candidates are being scored.
    """
    global _scored_cache, _original_data, _current_hypotheses, _current_weights
    logs = []
    def log(m): logs.append(str(m))
    def get_logs(): return "\n".join(logs)

    # Collect active hypotheses + weights from flat interleaved UI list
    hyps, weights = [], []
    for i in range(MAX_HYPOTHESES):
        h = hyp_weight_vals[i * 2]
        w = hyp_weight_vals[i * 2 + 1]
        if h and str(h).strip():
            hyps.append(str(h).strip())
            try:
                weights.append(float(w) if w else 1.0)
            except (TypeError, ValueError):
                weights.append(1.0)

    if not hyps:
        log("❌ No hypotheses found. Run Step 1 first.")
        yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
        return

    _current_hypotheses = hyps
    _current_weights    = weights

    yield get_logs(), None, None, None, render_progress_html(0, 1, label="Loading candidates…")

    if demo_mode:
        file_content = DEMO_CANDIDATES
        log("🎬 Demo Mode: 5 built-in candidates.")
    else:
        if not candidates_file:
            log("❌ Please upload a candidates file.")
            yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
            return
        try:
            fp = candidates_file if isinstance(candidates_file, str) else candidates_file.name
            with open(fp, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
        except Exception as ex:
            log(f"❌ Could not read candidates file: {ex}")
            yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
            return

    try:
        try:
            all_cands = json.loads(file_content)
            if not isinstance(all_cands, list):
                all_cands = [all_cands]
            log("✅ Parsed JSON format.")
        except json.JSONDecodeError:
            log("✅ Parsing JSONL format.")
            all_cands = [json.loads(l) for l in file_content.split("\n") if l.strip()]

        if not all_cands:
            log("❌ Candidates file parsed but contained no records.")
            yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
            return

        _original_data = all_cands
        parsed, skipped = [], 0
        for c in all_cands:
            cid = c.get("candidate_id")
            if not cid:
                skipped += 1
                continue
            profile = c.get("profile", {})
            history = c.get("career_history", [])
            blocks  = [profile.get("summary", "")]
            for job in history:
                blocks.append(
                    f"At {job.get('company','')} as {job.get('title','')}. "
                    f"{job.get('description','')}"
                )
            premise = " ".join(b for b in blocks if b).strip()
            if not premise:
                skipped += 1
                continue
            parsed.append({
                "candidate_id": cid,
                "premise":      premise,
                "signals":      clamp_signals(c.get("redrob_signals", {})),
                "title":        profile.get("current_title", "Engineer"),
                "yoe":          profile.get("years_of_experience", 0),
            })

        if skipped:
            log(f"⚠️  Skipped {skipped} record(s) — missing id or text.")
        log(f"✅ {len(parsed)} candidates loaded.")
        if not parsed:
            log("❌ No usable candidates after parsing.")
            yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
            return

    except Exception as ex:
        log(f"❌ Parse error: {ex}")
        yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
        return

    log("🧠 Loading NLI model…")
    yield get_logs(), None, None, None, render_progress_html(0, len(parsed), label="Loading model…")
    try:
        nli = get_nli_model()
    except Exception as ex:
        log(f"❌ Could not load NLI model: {ex}")
        yield get_logs(), None, None, None, render_progress_html(0, 0, error=True)
        return

    n_cands = len(parsed)
    log(f"🚀 Scoring {n_cands} candidates × {len(hyps)} hypotheses…")
    yield get_logs(), None, None, None, render_progress_html(0, n_cands)

    _scored_cache = {}
    for idx, cand in enumerate(parsed):
        try:
            result    = nli(cand["premise"][:2500], candidate_labels=hyps, multi_label=True)
            lbl2score = dict(zip(result["labels"], result["scores"]))
            ordered   = [max(0.0, min(1.0, lbl2score.get(h, 0.0))) for h in hyps]
        except Exception as ex:
            log(f"   ⚠️  NLI failed for {cand['candidate_id']}: {ex}")
            ordered = [0.0] * len(hyps)

        weighted_avg = (
            sum(s * w for s, w in zip(ordered, weights)) / sum(weights)
            if sum(weights) > 0 else 0.0
        )
        _scored_cache[cand["candidate_id"]] = {
            **cand,
            "nli_scores":       ordered,
            "nli_labels":       hyps,
            "nli_weights":      weights,
            "nli_weighted_avg": round(weighted_avg, 4),
            "red_flags":        detect_red_flags(cand["signals"], ordered),
        }

        # Yield a live progress update. Throttle slightly so we don't flood
        # the UI for very large candidate sets (update every candidate for
        # small sets, every Nth for big ones).
        step = max(1, n_cands // 200)  # at most ~200 UI updates total
        if (idx + 1) % step == 0 or (idx + 1) == n_cands:
            yield (
                get_logs(),
                None, None, None,
                render_progress_html(idx + 1, n_cands),
            )

    log("✅ NLI scoring complete.")
    yield get_logs(), None, None, None, render_progress_html(n_cands, n_cands, label="Finalizing…")

    log("🧮 Computing final scores with spread optimisation…")
    raw_finals = []
    for orig in _original_data:
        cid = orig.get("candidate_id")
        if cid not in _scored_cache:
            continue
        sc  = _scored_cache[cid]
        sig = sc["signals"]
        rr  = sig.get("recruiter_response_rate", 0.5)
        gh  = sig.get("github_activity_score", 0) / 100.0
        beh = (rr + gh) / 2.0
        raw_finals.append((cid, sc["nli_weighted_avg"] * 0.60 + beh * 0.40))

    if not raw_finals:
        log("❌ No candidates could be scored.")
        yield get_logs(), None, None, None, render_progress_html(n_cands, n_cands, error=True)
        return

    cids_order = [x[0] for x in raw_finals]
    raw_scores = [x[1] for x in raw_finals]
    stretched  = apply_score_stretch(raw_scores, power=1.5)
    pct_ranks  = percentile_ranks(raw_scores)

    final = []
    for cid, raw, s_score, pct in zip(cids_order, raw_scores, stretched, pct_ranks):
        sc = _scored_cache[cid]
        sc["final_score"]     = round(s_score, 4)
        sc["raw_score"]       = round(raw, 4)
        sc["percentile_rank"] = round(pct, 1)
        final.append({"candidate_id": cid, "final_score": s_score, "cand": sc})

    final.sort(key=lambda x: x["final_score"], reverse=True)
    top100 = final[:100]

    # --- CSV EXPORT (Plain Text Only) ---
    csv_path = os.path.join(tempfile.gettempdir(), "team_submission.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top100, 1):
            c = item["cand"]
            formatted_score_csv = f"{item['final_score']:.4f} ({c['raw_score']:.4f})"
            w.writerow([
                item["candidate_id"],
                rank,
                formatted_score_csv,
                build_exec_summary(c, as_html=False), # Keeps CSV clean
            ])
    log(f"💾 CSV exported ({len(top100)} candidates) — columns: candidate_id, rank, score, reasoning.")

    # --- UI TABLE (HTML/Markdown Styled) ---
    table_rows = []
    for rank, item in enumerate(top100[:10], 1):
        c = item["cand"]
        # Score column gets a subtle color treatment
        formatted_score_html = f"<b style='color:#58a6ff'>{item['final_score']:.4f}</b> <br><span style='color:#8b949e'>({c['raw_score']:.4f})</span>"
        table_rows.append([
            item["candidate_id"],
            rank,
            formatted_score_html,
            build_exec_summary(c, as_html=True), # Injects CSS styling and <br> tags
        ])

    radar_data = {}
    for item in top100[:5]:
        c = item["cand"]
        radar_data[c["candidate_id"]] = {
            "scores": c["nli_scores"],
            "labels": [f"H{i+1}" for i in range(len(c["nli_scores"]))],
            "final":  item["final_score"],
            "flags":  c["red_flags"],
            "pct":    c.get("percentile_rank", 0),
        }

    log("🎉 Done! Review the Top 10 below and download the CSV.")
    yield (
        get_logs(),
        csv_path,
        table_rows,
        json.dumps(radar_data),
        render_progress_html(n_cands, n_cands, done=True),
    )


# ==========================================
# Radar HTML
# ==========================================
def make_radar_html(radar_json):
    if not radar_json:
        return "<p style='color:#8b949e;text-align:center;padding:40px'>Run scoring to see radar charts.</p>"
    try:
        data = json.loads(radar_json)
    except Exception:
        return "<p style='color:#da3633'>Could not parse radar data.</p>"
    if not data:
        return "<p style='color:#8b949e;text-align:center;padding:40px'>No candidates to display.</p>"

    palette = ["#58a6ff", "#3fb950", "#bc8cff", "#f78166", "#e3b341"]
    cards   = ""
    for i, (cid, info) in enumerate(data.items()):
        scores = info["scores"]
        labels = info["labels"]
        flags  = info.get("flags", [])
        final  = info.get("final", 0)
        pct    = info.get("pct", 0)
        color  = palette[i % len(palette)]
        n      = len(scores)
        if n == 0:
            continue

        flag_html = " ".join(
            f"<span style='background:#da3633;color:#fff;font-size:.65rem;"
            f"border-radius:3px;padding:1px 5px;margin-left:4px'>{f}</span>"
            for f in flags
        )
        cx = cy = 100; r = 70
        pts  = [(cx + r * s * math.cos(math.radians(90 - j * 360 / n)),
                 cy - r * s * math.sin(math.radians(90 - j * 360 / n)))
                for j, s in enumerate(scores)]
        grid = "".join(
            f'<circle cx="{cx}" cy="{cy}" r="{r*g/4:.1f}" fill="none" '
            f'stroke="#30363d" stroke-width="0.5"/>'
            for g in range(1, 5)
        )
        axes = lbrs = ""
        for j, lbl in enumerate(labels):
            ang = math.radians(90 - j * 360 / n)
            x2, y2 = cx + r * math.cos(ang),       cy - r * math.sin(ang)
            xl, yl = cx + (r+14) * math.cos(ang),  cy - (r+14) * math.sin(ang)
            axes += (f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#30363d" stroke-width="0.8"/>')
            lbrs += (f'<text x="{xl:.1f}" y="{yl:.1f}" fill="#8b949e" font-size="7" '
                     f'text-anchor="middle" dominant-baseline="middle">{lbl}</text>')
        poly = (f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" '
                f'fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="1.5"/>')
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
            for x, y in pts
        )
        svg = (f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" '
               f'style="width:160px;height:160px">{grid}{axes}{poly}{dots}{lbrs}</svg>')

        cards += (
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
            f'padding:14px;min-width:200px;flex-shrink:0">'
            f'<div style="font-size:.78rem;font-weight:700;color:{color};margin-bottom:2px">'
            f'{cid}{flag_html}</div>'
            f'<div style="font-size:.68rem;color:#8b949e;margin-bottom:6px">'
            f'Score: <b style="color:{color}">{final:.3f}</b>'
            f'&nbsp;·&nbsp;<b style="color:#e3b341">{pct:.0f}th %ile</b>'
            f'</div>{svg}</div>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;gap:12px;padding:8px">{cards}</div>'


# ==========================================
# CSS
# ==========================================
CSS = """
body, .gradio-container { background:#0d1117 !important; font-family:'Inter',sans-serif; }
.app-header { text-align:center; padding:28px 0 12px; }
.app-header h1 { font-size:2rem; font-weight:700;
  background:linear-gradient(135deg,#58a6ff,#bc8cff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
.app-header p { color:#8b949e; margin-top:6px; font-size:.95rem; }
.section-label { font-size:.75rem; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; color:#58a6ff; margin-bottom:10px; }
.log-box textarea { background:#0d1117 !important; color:#3fb950 !important;
  font-family:'JetBrains Mono',monospace !important; font-size:.8rem !important;
  border:1px solid #30363d !important; border-radius:8px !important; }
.gr-dataframe table { background:#161b22; color:#c9d1d9; font-size:.82rem; }
.gr-dataframe th { background:#21262d; color:#58a6ff; }
"""

# ==========================================
# Gradio UI
# ==========================================
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as app:

    gr.HTML("""
    <div class="app-header">
      <h1>🚀 AI Hiring Pipeline</h1>
      <p>Gemini hypothesis extraction · NLI scoring · Weighted ranking · Radar analytics</p>
    </div>
    """)

    # ── Control bar ───────────────────────────────────────────────────────────
    with gr.Row():
        demo_toggle = gr.Checkbox(label="🎬 Demo Mode (no files needed)", value=False, scale=2)
        api_key_in  = gr.Textbox(label="Gemini API Key", type="password",
                                  placeholder="AIzaSy…", scale=4)
        num_hyp_in  = gr.Slider(minimum=3, maximum=MAX_HYPOTHESES, value=5, step=1,
                                 label="Hypothesis Count", scale=2)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="padding:0 8px">'
            'Step 1 — Job Description &amp; Hypotheses</div>')
    with gr.Row():
        with gr.Column(scale=3):
            jd_file_in = gr.File(label="Job Description (.txt / .json)", type="filepath")
        with gr.Column(scale=1):
            extract_btn    = gr.Button("⚡ Extract Hypotheses", variant="primary")
            regenerate_btn = gr.Button("🔄 Regenerate")

    extract_log = gr.Textbox(label="Step 1 Log", lines=5, interactive=False,
                              elem_classes="log-box")

    # ── Hypothesis rows ───────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="padding:0 8px;margin-top:8px">'
            'Hypotheses — edit freely before scoring</div>')

    hyp_rows       = []   # gr.Row     × MAX_HYPOTHESES
    hyp_boxes      = []   # gr.Textbox × MAX_HYPOTHESES
    weight_sliders = []   # gr.Slider  × MAX_HYPOTHESES

    for i in range(MAX_HYPOTHESES):
        with gr.Row(visible=False) as row:
            gr.HTML(f'<div style="color:#58a6ff;font-weight:700;font-size:.85rem;'
                    f'min-width:24px;padding-top:8px">{i+1}</div>')
            hb = gr.Textbox(
                placeholder=f"Hypothesis {i+1}…",
                show_label=False, scale=6,
                interactive=True, container=False,
            )
            ws = gr.Slider(
                minimum=0.1, maximum=3.0, value=1.0, step=0.1,
                label="Weight", scale=2, info="1.0 = baseline",
            )
        hyp_rows.append(row)
        hyp_boxes.append(hb)
        weight_sliders.append(ws)

    # ── Edit instruction ──────────────────────────────────────────────────────
    with gr.Row():
        edit_box       = gr.Textbox(
            label="✏️ Edit instruction (optional — appended to prompt before re-extracting)",
            placeholder="E.g. focus more on Kubernetes experience…",
            interactive=True, lines=1, scale=5,
        )
        apply_edit_btn = gr.Button("Apply Edit & Re-Extract", scale=1)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="padding:0 8px;margin-top:16px">'
            'Step 2 — Score Candidates</div>')
    with gr.Row():
        candidates_file_in = gr.File(label="Candidates (.json / .jsonl)",
                                      type="filepath", scale=3)
        run_btn = gr.Button("▶ Run Scoring", variant="primary", scale=1)

    score_log = gr.Textbox(label="Step 2 Log (Errors & Summary)", lines=6, interactive=False,
                            elem_classes="log-box")

    # ── Results ───────────────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="padding:0 8px;margin-top:16px">'
            'Results — Top 10</div>')

    # Live, centered progress block — sits above the results table and
    # updates continuously while run_scoring() is yielding.
    progress_html_out = gr.HTML(render_progress_html(0, 0))

    gr.HTML(
        "<p style='color:#8b949e;font-size:.78rem;padding:0 8px;margin:-4px 0 8px'>"
        "Output columns match the required submission format: "
        "<b style='color:#58a6ff'>candidate_id · rank · score · reasoning</b>. "
        "Scores are <b style='color:#e3b341'>Normalized (Raw)</b>.</p>"
    )

    # Changed datatype to "markdown" so it renders the injected HTML tags for coloring
    top10_table = gr.Dataframe(
        headers=["candidate_id", "rank", "score", "reasoning"],
        datatype=["str", "number", "markdown", "markdown"],
        interactive=False, wrap=True,
    )
    csv_dl = gr.File(label="⬇ Download Full CSV (Top 100)")

    # ── Radar ─────────────────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="padding:0 8px;margin-top:16px">'
            'Radar — Top 5 Hypothesis Profiles</div>')
    radar_html_out   = gr.HTML(
        "<p style='color:#8b949e;text-align:center;padding:40px'>"
        "Run scoring to see radar charts.</p>"
    )
    radar_data_state = gr.State("")

    # ── Demo toggle ───────────────────────────────────────────────────────────
    demo_toggle.change(
        fn=lambda d: (gr.update(visible=not d), gr.update(visible=not d)),
        inputs=[demo_toggle],
        outputs=[jd_file_in, candidates_file_in],
    )

    # ── Extract wiring ────────────────────────────────────────────────────────
    all_extract_outputs = [extract_log] + hyp_rows + hyp_boxes + weight_sliders

    def on_extract(api_key, jd_file, n_hyp, demo, edit=""):
        log_str, hyps = extract_hypotheses(api_key, jd_file, n_hyp, demo, edit)
        n = len(hyps)

        row_updates = [gr.update(visible=(i < n))                              for i in range(MAX_HYPOTHESES)]
        tb_updates  = [gr.update(value=hyps[i] if i < n else "", visible=(i < n)) for i in range(MAX_HYPOTHESES)]
        sl_updates  = [gr.update(value=1.0,                        visible=(i < n)) for i in range(MAX_HYPOTHESES)]

        return [log_str] + row_updates + tb_updates + sl_updates

    extract_btn.click(
        fn=on_extract,
        inputs=[api_key_in, jd_file_in, num_hyp_in, demo_toggle],
        outputs=all_extract_outputs,
    )
    regenerate_btn.click(
        fn=on_extract,
        inputs=[api_key_in, jd_file_in, num_hyp_in, demo_toggle],
        outputs=all_extract_outputs,
    )
    apply_edit_btn.click(
        fn=on_extract,
        inputs=[api_key_in, jd_file_in, num_hyp_in, demo_toggle, edit_box],
        outputs=all_extract_outputs,
    )

    # ── Scoring wiring ────────────────────────────────────────────────────────
    all_hyp_weight_inputs = []
    for hb, ws in zip(hyp_boxes, weight_sliders):
        all_hyp_weight_inputs += [hb, ws]

    def on_run_scoring(cands_file, demo, *hyp_weight_vals):
        """
        Streams (score_log, csv_dl, top10_table, radar_html_out, radar_data_state)
        updates as run_scoring() yields. Intermediate steps leave the table/csv
        untouched (gr.update()) and only refresh the progress HTML; the final
        yield fills in the real table, csv path, and radar chart.
        """
        for logs, csv_path, table, radar_json, progress_html in run_scoring(
            cands_file, demo, hyp_weight_vals
        ):
            table_update = table if table is not None else gr.update()
            csv_update   = csv_path if csv_path is not None else gr.update()

            if radar_json is not None:
                radar_update       = make_radar_html(radar_json)
                radar_state_update = radar_json
            else:
                radar_update       = gr.update()
                radar_state_update = gr.update()

            yield logs, csv_update, table_update, radar_update, radar_state_update, progress_html

    run_btn.click(
        fn=on_run_scoring,
        inputs=[candidates_file_in, demo_toggle] + all_hyp_weight_inputs,
        outputs=[score_log, csv_dl, top10_table, radar_html_out, radar_data_state, progress_html_out],
    )


if __name__ == "__main__":
    app.queue()
    app.launch()
