# -*- coding: utf-8 -*-
# ===========================================================
# 🚗 AI Deal Benchmark — 25 Slots × 2 Rounds
# Gemini 2.5 Pro (Tool prompt, web reasoning) vs GPT-4o (knowledgeable buyer)
# Per-slot "Analyze" buttons • Auto-Save (always-on download) • Resume • Planned stops @12/@25 each round
# ===========================================================

import os, json, time, re
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from json_repair import repair_json

# APIs
import google.generativeai as genai
from openai import OpenAI

# ------------------------- CONFIG ---------------------------
APP_VERSION = "2.2.0"
st.set_page_config(page_title="AI Deal Benchmark 25", page_icon="🚗", layout="wide")
st.title("🚗 AI Deal Benchmark — 25 Listings × 2 Rounds")
st.caption(f"Gemini 2.5 Pro (Tool) vs GPT-4o (Buyer) • Per-slot Analyze • Auto-Save/Resume • Planned Stops • v{APP_VERSION}")

# Secrets
GEMINI_KEY  = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_KEY  = st.secrets.get("OPENAI_API_KEY", "")
if not GEMINI_KEY or not OPENAI_KEY:
    st.error("Missing GEMINI_API_KEY or OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# Init clients
genai.configure(api_key=GEMINI_KEY)
gem_model = genai.GenerativeModel("gemini-2.5-pro")
gpt_client = OpenAI(api_key=OPENAI_KEY)

# Files
AUTOSAVE_CSV     = "autosave.csv"
FINAL_ROUND_CSV  = "results_round{r}.csv"
FINAL_COMBINED   = "results_combined.csv"

# Planned stops
PLANNED_STOPS = {1: {12, 25}, 2: {12, 25}}

# ----------------------- HELPERS ----------------------------
def parse_json_safe(raw: str):
    raw = (raw or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(repair_json(raw))
        except Exception:
            return {}

def img_to_jpeg_bytes(file):
    if not file:
        return None
    try:
        img = Image.open(file).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None

def ensure_df_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "round","slot","ad_text",
        "gemini_json","gpt_text",
        "eval_winner","eval_gemini_acc","eval_gpt_acc","eval_summary",
        "timestamp"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = []
    return df[cols]

def load_autosave() -> pd.DataFrame:
    if os.path.exists(AUTOSAVE_CSV):
        try:
            df = pd.read_csv(AUTOSAVE_CSV)
            return ensure_df_schema(df)
        except Exception:
            pass
    return ensure_df_schema(pd.DataFrame())

def save_autosave(df: pd.DataFrame):
    df = ensure_df_schema(df)
    df.to_csv(AUTOSAVE_CSV, index=False, encoding="utf-8")

def download_df(df: pd.DataFrame, label="📥 Download Latest Auto-Save", key_suffix=None):
    df = ensure_df_schema(df)
    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        unique_key = f"download_{key_suffix or int(time.time()*1000)}"
        st.download_button(
            label,
            csv_data,
            file_name="autosave.csv",
            mime="text/csv",
            use_container_width=True,
            key=unique_key
        )

def resume_progress(df: pd.DataFrame):
    """Detect next round+slot to continue from."""
    if df is None or df.empty:
        return 1, 1
    last = df.sort_values(["round","slot"]).tail(1)
    last_round = int(last["round"].values[0])
    last_slot  = int(last["slot"].values[0])
    if last_slot < 25:
        return last_round, last_slot + 1
    else:
        return min(2, last_round + 1), 1

def build_extra(vin, zip_code, seller, photo_count):
    parts = []
    if vin: parts.append(f"VIN: {vin}")
    if zip_code: parts.append(f"ZIP/State: {zip_code}")
    if seller: parts.append(f"Seller: {seller}")
    if photo_count: parts.append(f"Photos provided: {photo_count}")
    return "\n".join(parts)

# ----------------------- PROMPTS (YOUR EXACT TEXT) ----------------------------
# 1) FULL TOOL PROMPT (Gemini) — EXACTLY YOUR PROMPT, as provided
def build_gemini_tool_prompt(ad: str, extra: str, must_id: str, exact_json: dict, similar_json: list) -> str:
    exact_s = json.dumps(exact_json or {}, ensure_ascii=False)
    similar_s = json.dumps(similar_json or [], ensure_ascii=False)
    return f"""
You are a senior U.S. used-car analyst (2023–2025). Web reasoning is REQUIRED.

Stages:
1) Extract listing facts: ask_price_usd, brand, model, year, trim, powertrain, miles, title_status, owners, accidents,
   options_value_usd, state_or_zip, days_on_market (if present).
2) Do live U.S.-centric lookups (REQUIRED) for the exact year/model:
   - Market comps & CLEAN-title median: Cars.com, Autotrader, Edmunds, and KBB (Kelley Blue Book).
   - Reliability & common issues: Consumer Reports style + RepairPal.
   - Typical annual maintenance cost: RepairPal or YourMechanic (U.S. 2023–2025).
   - Depreciation trend (24–36m): CarEdge or iSeeCars.
   - Demand/DOM averages; brand/model resale retention (CarEdge/iSeeCars).
   - Safety/recalls context: NHTSA; insurance risk context: IIHS (qualitative).
   - Verify warranty status via manufacturer website; if warranty expired, lower reliability and raise failure-risk weighting accordingly and explain it in the reliability section.
   - Verify open recalls and TSBs via NHTSA/manufacturer; check lemon-law/buyback if VIN present.
   Consider U.S. realities (Rust Belt vs Sun Belt, dealer vs private, mileage normalization).

Use prior only for stabilization (do NOT overfit):
- exact_prev (same listing id): weight ≤ 25% -> {exact_s}
- similar_previous (very similar ads): anchors only, weight ≤ 10% -> {similar_s}

Scoring rules for U.S. buyers (adjusted weights):
- Title condition (clean > rebuilt > salvage) ~20%; if 'rebuilt'/'salvage'/branded -> CAP deal_score ≤ 75.
- Price vs CLEAN-title median ~25%.
- Mileage impact ~10% (U.S. highway-heavy driving reduces penalty).
- Reliability & maintenance together ~20%.
- TCO (fuel + insurance + repairs) ~8% (U.S. costs).
- Accidents + owners ~9%.
- Rust/flood zone ~4% (Rust Belt/coastal exposure).
- Demand/resale ~4%.

Critical adjustment guidelines (U.S.-market realism):
Edge-case heuristic layer (20 scenarios — apply in addition to base weights):
1) OEM new engine → Reliability +25–35; Market +15; Resale +10.
2) Used/unknown-provenance engine → ≤ +5; add caution flag (“verify installation origin”).
3) OEM new transmission → Reliability +15; Market +10.
4) Rebuilt / Salvage / Branded title → cap deal_score ≤ 75; ROI_expected −5.
5) Carfax “minor damage” → −5 reliability; −5 resale (acceptable if repaired).
6) Structural damage / airbag deployed → set ceiling ≤ 55 overall; strong warning.
7) Repainted panels / full repaint → −5 market; −5 resale.
8) Clean Carfax + 1 owner + dealer maintained → +10 reliability; +10 resale.
9) High-insurance states (MI, NY, NJ, FL) → −5 TCO; mention insurance context.
10) Sun Belt (FL, AZ, CA, TX, NV) → +5 rust; −2 interior (sun wear) if hinted.
11) Rust Belt origin/operation → −10 rust; add underbody inspection warning.
12) Suspiciously low miles for age with no documentation → −10 reliability until explained.
13) Fleet/Rental history → −10 reliability; −10 resale.
14) Private owner + full service records → +10 reliability; +5 resale.
15) High-performance trims (AMG/M/M S-line/Hellcat) → +10 demand/market; −5 TCO (insurance).
16) Extensive aftermarket mods/tuning → −10 resale; −5 reliability (unless track-documented).
17) Canada-import / grey market → −10 market; −10 resale; mention potential registration/insurance frictions.
18) Major recall fixed with proof → +5 reliability.
19) Hybrid/EV traction battery recently replaced → +20 reliability; +10 resale.
Extended risk and compliance logic (add below the 20 edge cases):
• Always cross-check safety recalls via NHTSA and active TSBs; if open recalls found, reduce reliability −5 and include note.
• If VIN indicates manufacturer buyback or lemon law history → cap deal_score ≤ 65 and flag as "Lemon/Buyback risk".
• If odometer discrepancy or title mileage not actual → cap ≤ 60 and mention "Not actual mileage".
• If warranty expired (<5yr/60k mi for mainstream, <4yr/50k for luxury) → −10 reliability, −5 resale; explain in reliability rationale.
• If factory powertrain warranty active → +10 reliability confidence.
• If Carfax shows “commercial use” (fleet, rental, ride-share) → −10 reliability, −10 resale; warn about heavy wear.
• If service records show all-dealer maintenance → +10 reliability, +5 resale.
• If listing location is in flood-prone ZIP (Louisiana, Florida coastal, Texas Gulf) → −8 rust/flood; mention flood risk explicitly.
• If ad mentions “as-is sale”, “no warranty”, or “mechanic special” → reduce confidence and market valuation significantly (−10 to −20 combined).
• If luxury performance (AMG/M/Hellcat) and tire/maintenance costs > $2k/year → −10 TCO, note high cost-of-ownership.
• If EV with degraded or replaced battery (verified via manufacturer) → adjust reliability ±20 depending on replacement status.
• If EV range <80% of original → cap deal_score ≤ 70 and mention “battery degradation”.
• Always ensure numeric consistency: explanation text must never contradict any component score.
• End each explanation with a short ROI summary: expected return (12/24/36m) and key U.S. buyer takeaway.
20) “As-is” sale with no warranty → −10 confidence; −10 resale; emphasize PPI.
• If listing text mentions any of these keywords:
  ["new engine", "engine replaced", "factory engine replaced", "rebuilt transmission", "new transmission", "engine under warranty", "factory rebuild", "powertrain warranty", "short block replaced"]
  → Apply a strong positive adjustment:
    - Reliability: +25 to +35 points
    - Mileage: +20 points
    - Market: +15 points
    - Resale_value: +10 points
    - Add explicit note in "score_explanation" about OEM/warranty-grade replacement.
• If replacement keywords appear without "OEM", "warranty", or "dealer-installed":
  → Moderate/neutral (+10–15 total) and flag provenance uncertainty.
• Align numeric component scores with narrative (no contradictions).

Edge-case heuristic layer (20 scenarios — apply in addition to base weights).

Explanation contract (MANDATORY):
- Return a specific, human-readable explanation tying PRICE vs CLEAN median, TITLE, MILEAGE, RELIABILITY/MAINTENANCE (with U.S. sources), warranty status, and ROI.
- 120–400 words, 3–6 bullets/short paragraphs.
- Mention at least two anchors by name (KBB, Edmunds, RepairPal, iSeeCars, etc.).
- DO NOT copy any instruction text or placeholders.

Output STRICT JSON only:
{{
  "from_ad": {{"brand":"","model":"","year":null,"vin":"","seller_type":""}},
  "ask_price_usd": 0,
  "vehicle_facts": {{
    "title_status":"unknown","accidents":0,"owners":1,"dealer_reputation":null,
    "rarity_index":0,"options_value_usd":0,"days_on_market":0,"state_or_zip":"","miles":null
  }},
  "market_refs": {{"median_clean":0,"gap_pct":0}},
  "web_search_performed": true,
  "confidence_level": 0.75,
  "components": [
    {{"name":"market","score":0,"note":""}},
    {{"name":"title","score":0,"note":""}},
    {{"name":"mileage","score":0,"note":""}},
    {{"name":"reliability","score":0,"note":""}},
    {{"name":"maintenance","score":0,"note":""}},
    {{"name":"tco","score":0,"note":""}},
    {{"name":"accidents","score":0,"note":""}},
    {{"name":"owners","score":0,"note":""}},
    {{"name":"rust","score":0,"note":""}},
    {{"name":"demand","score":0,"note":""}},
    {{"name":"resale_value","score":0,"note":""}}
  ],
  "deal_score": 0,
  "roi_forecast_24m": {{"expected":0,"optimistic":0,"pessimistic":0}},
  "roi_forecast": {{"12m":0,"24m":0,"36m":0}},
  "risk_tier": "Tier 2 (average-risk)",
  "relative_rank": "",
  "buyer_fit": "",
  "verification_summary": "",
  "benchmark": {{"segment":"","rivals":[]}},
  "score_explanation": "<<WRITE DETAILED EXPLANATION — NO PLACEHOLDERS>>",
  "listing_id_used": "{must_id}"
}}

LISTING (title + description):
\"\"\"{ad}\"\"\"
Extra:
{extra}

Hard constraints:
- Always perform web lookups and set web_search_performed=true; if not possible, list which sources failed but still estimate.
- Numeric fields must be numbers. deal_score: 0..100. ROI parts: -50..50.
- Per-component short notes required.
- If title_status is 'rebuilt', 'salvage' or any branded title: CAP deal_score ≤ 75 and clearly warn in score_explanation.
- If market gap (gap_pct) ≤ -35: warn to verify insurance/accident history before purchase.
- Enforce alignment between narrative and component scores (no contradictions).
"""

# 2) GPT PROMPT — knowledgeable buyer (EN)
def build_gpt_prompt(ad_text: str) -> str:
    return f"""
You are a knowledgeable U.S. used-car buyer with strong grasp of reliability, insurance, depreciation, and resale.
You routinely check KBB/Edmunds/CarEdge and read RepairPal/Consumer Reports before buying.

Listing:
\"\"\"{ad_text}\"\"\"

Write a concise, realistic buyer’s take:
- Price fairness vs U.S. market norms.
- Red flags (title/accidents/mileage/state/insurance exposure/gray-market/etc.).
- Brand- and model-specific pros/cons.
- Resale outlook and ownership cost view (1–2 lines).
- Final gut verdict (“good deal if PPI”, “borderline”, “avoid/overpriced”).

Respond in 2–4 short paragraphs. No fluff, no placeholders.
"""

# 3) EVALUATOR PROMPT — strict JSON + fact-check pressure
def build_evaluator_prompt(ad_text: str, gem_out: str, gpt_out: str) -> str:
    return f"""
You are a neutral U.S. automotive fact-checker with REQUIRED web reasoning.
Compare two analyses of the same listing and verify factual accuracy using U.S. anchors:
KBB, Edmunds, Cars.com, Autotrader, RepairPal, iSeeCars/CarEdge, NHTSA, IIHS.

Scoring rubric (accuracy focus):
- Market alignment (price vs clean-title median) and numeric consistency.
- Correct handling of title/mileage/accidents/warranty.
- Proper U.S. cost/reliability references (RepairPal, Consumer Reports style).
- Evidence of live web verification (penalize if absent).
Return STRICT JSON only:
{{
  "gemini_accuracy_score": 0,
  "gpt_accuracy_score": 0,
  "winner": "Gemini" | "GPT" | "Tie",
  "evaluation_summary": "Short explanation mentioning at least one source and any contradictions found."
}}

LISTING:
\"\"\"{ad_text[:1800]}\"\"\"

GEMINI ANALYSIS (tool JSON or text):
\"\"\"{(gem_out or '')[:1800]}\"\"\"

GPT ANALYSIS (buyer-style text):
\"\"\"{(gpt_out or '')[:1800]}\"\"\"
"""

# -------------------- MODEL CALLS ---------------------------
def call_gemini_tool(ad_text: str, extra: str, slot_id: str, exact_prev: dict=None, similar_list: list=None, img_bytes=None):
    parts = [{"text": build_gemini_tool_prompt(ad_text, extra, slot_id, exact_prev or {}, similar_list or [])}]
    if img_bytes:
        parts.append({"mime_type":"image/jpeg","data":img_bytes})
    r = gem_model.generate_content(parts, request_options={"timeout": 180})
    return r.text or ""

def call_gpt_buyer(ad_text: str):
    comp = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": build_gpt_prompt(ad_text)}],
        temperature=0.5,
        max_tokens=800,
    )
    return comp.choices[0].message.content

def call_evaluator(ad_text: str, gem_out: str, gpt_out: str):
    r = gem_model.generate_content([{"text": build_evaluator_prompt(ad_text, gem_out, gpt_out)}],
                                   request_options={"timeout": 180})
    return parse_json_safe(r.text or "{}")

# ------------------------- UI -------------------------------
st.sidebar.header("Run Control")

mode = st.sidebar.radio("Mode", ["New run", "Resume from file"], horizontal=True)
autosave_df = load_autosave()

if mode == "Resume from file":
    f = st.sidebar.file_uploader("Upload autosave CSV", type=["csv"])
    if f:
        try:
            autosave_df = pd.read_csv(f)
            autosave_df = ensure_df_schema(autosave_df)
            st.sidebar.success(f"Loaded {len(autosave_df)} rows from uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

# Sidebar live download button — always available
st.sidebar.markdown("**Auto-Save (always latest):**")
download_df(autosave_df, key_suffix="sidebar")

auto_round, auto_slot = resume_progress(autosave_df)
round_choice = st.sidebar.selectbox("Round", [1,2], index=(auto_round-1))
st.sidebar.caption(f"Auto-detected next slot: round {auto_round}, slot {auto_slot}")

# Global metadata (optional)
st.markdown("### Global details (optional, applied to all slots on analyze)")
g1, g2, g3 = st.columns(3)
with g1:
    vin_global  = st.text_input("VIN", "")
with g2:
    zip_global  = st.text_input("ZIP / State", "")
with g3:
    seller_type = st.selectbox("Seller", ["","private","dealer"])

# Inputs grid — 25 slots
st.markdown("### Enter up to 25 real listings (text + optional image)")
slot_inputs = []
grid_cols = st.columns(5)
for i in range(25):
    col = grid_cols[i % 5]
    with col.expander(f"Slot #{i+1}", expanded=(i<5)):
        txt = st.text_area(f"Listing text #{i+1}", key=f"txt_{i+1}", height=160, placeholder="Year • Make • Model • Trim • Mileage • Price • Title • Location • Options …")
        img = st.file_uploader(f"Image #{i+1}", type=["jpg","jpeg","png"], key=f"img_{i+1}")
        # Per-slot analyze button
        slot_inputs.append((txt, img))

# Always-on download button under grid
download_df(autosave_df, label="📥 Download Auto-Save (below grid)", key_suffix="below_grid")

# --------------------- PER-SLOT ANALYZE ---------------------
st.markdown("---")
st.subheader("Per-slot Analyze (Gemini vs GPT)")

results = autosave_df.copy()

# Show 25 buttons in a 5×5 grid: “Analyze Slot #i”
btn_cols = st.columns(5)
for i in range(25):
    col = btn_cols[i % 5]
    with col:
        if st.button(f"Analyze #{i+1}", key=f"analyze_{round_choice}_{i+1}"):
            ad_text, img_file = slot_inputs[i]
            slot_id = i + 1

            if not (ad_text and ad_text.strip()):
                st.warning(f"Slot {slot_id}: Please paste listing text first.")
                st.stop()

            # Check if this slot already processed in this round
            already = False
            if not results.empty:
                already = ((results["round"]==round_choice) & (results["slot"]==slot_id)).any()
            if already:
                st.info(f"Round {round_choice} Slot {slot_id} already exists in autosave. Replacing with new analysis…")
                results = results[~((results["round"]==round_choice) & (results["slot"]==slot_id))].copy()

            img_bytes = img_to_jpeg_bytes(img_file)
            must_id = f"R{round_choice}-S{slot_id}-{int(time.time())}"
            extra = build_extra(vin_global, zip_global, seller_type, 1 if img_bytes else 0)

            # (Basic memory anchors could be added here if desired; keeping simple for per-slot on-demand run)
            exact_prev, sims = {}, []

            with st.spinner(f"Gemini 2.5 Pro analyzing slot {slot_id}…"):
                gem_raw = call_gemini_tool(ad_text, extra, must_id, exact_prev, sims, img_bytes)
            gem_raw_trunc = (gem_raw or "")[:18000]

            with st.spinner(f"GPT-4o analyzing slot {slot_id}…"):
                gpt_text = call_gpt_buyer(ad_text)
            gpt_text_trunc = (gpt_text or "")[:18000]

            with st.spinner("Evaluator verifying…"):
                ev = call_evaluator(ad_text, gem_raw_trunc, gpt_text_trunc)
            ev_gem = int(ev.get("gemini_accuracy_score", 0))
            ev_gpt = int(ev.get("gpt_accuracy_score", 0))
            ev_w   = str(ev.get("winner","")).strip() or ("Gemini" if ev_gem>ev_gpt else ("GPT" if ev_gpt>ev_gem else "Tie"))
            ev_s   = str(ev.get("evaluation_summary","")).strip()

            new_row = {
                "round": round_choice,
                "slot": slot_id,
                "ad_text": ad_text[:2000],
                "gemini_json": gem_raw_trunc,
                "gpt_text": gpt_text_trunc,
                "eval_winner": ev_w,
                "eval_gemini_acc": ev_gem,
                "eval_gpt_acc": ev_gpt,
                "eval_summary": ev_s,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            results = pd.concat([ensure_df_schema(results), pd.DataFrame([new_row])], ignore_index=True)

            # Persist autosave immediately
            save_autosave(results)
            st.success(f"Saved Round {round_choice} Slot {slot_id} ✅")
            download_df(results, key_suffix=f"slot_{round_choice}_{slot_id}")

            # Planned stop logic
            if slot_id in PLANNED_STOPS.get(round_choice, set()):
                # Save round file and stop
                ensure_df_schema(results[results["round"] == round_choice]).to_csv(FINAL_ROUND_CSV.format(r=round_choice), index=False)
                st.warning(f"⏸ Planned stop after Round {round_choice} Slot {slot_id}. Download autosave.csv and resume later.")
                download_df(results, key_suffix=f"pause_{round_choice}_{slot_id}")
                st.stop()

# --------------------- SUMMARY & EXPORTS ---------------------
st.markdown("---")
st.subheader("Run Status")

# Show table preview for current round
round_df = ensure_df_schema(results[results["round"] == round_choice].copy())
st.dataframe(round_df.sort_values("slot"), use_container_width=True)

# Save round CSV and combined CSV buttons
c1, c2, c3 = st.columns(3)
with c1:
    if st.button(f"💾 Save Round {round_choice} CSV", key=f"save_round_{round_choice}"):
        ensure_df_schema(round_df).to_csv(FINAL_ROUND_CSV.format(r=round_choice), index=False)
        st.success(f"Saved {FINAL_ROUND_CSV.format(r=round_choice)}")
with c2:
    # Combine all rounds (if both exist)
    if st.button("🧩 Save Combined CSV (all)", key="save_combined"):
        ensure_df_schema(results).to_csv(FINAL_COMBINED, index=False)
        st.success(f"Saved {FINAL_COMBINED}")
with c3:
    download_df(results, label="📥 Download Auto-Save (footer)", key_suffix="footer")

st.caption("Tip: In case of unexpected crash, the latest autosave.csv is already on disk. Use 'Resume from file' to continue exactly where you left off.")
st.caption("AI Deal Benchmark 25 © 2025 — Gemini 2.5 Pro vs GPT-4o • Auto-Save/Resume • Per-slot Analyze • Planned Stops")
