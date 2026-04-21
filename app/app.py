import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import gdown

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Big Five Personality Test",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS  — เหมือน version เดิม แก้แค่ text สว่างขึ้น
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 700px; }
.stProgress > div > div { background: #7F77DD; border-radius: 99px; }
.stProgress > div { background: #ececec; border-radius: 99px; }
div[data-testid="column"] button {
    width:100%!important; border-radius:12px!important;
    font-size:1.2rem!important; font-weight:500!important; height:56px!important;
    border:1.5px solid #e0e0e0!important; background:white!important;
    color:#333!important; transition:all 0.12s ease!important;
}
div[data-testid="column"] button:hover {
    border-color:#7F77DD!important; color:#534AB7!important;
    background:#f5f4ff!important; transform:translateY(-2px)!important;
}
.stButton > button[kind="primary"] {
    background:#7F77DD!important; color:white!important; border:none!important;
    border-radius:12px!important; font-weight:500!important;
    font-size:15px!important; padding:0.65rem 1.5rem!important;
}
.stButton > button[kind="primary"]:hover { background:#534AB7!important; }
.stButton > button[kind="secondary"] {
    color:#555!important; border-color:#ddd!important;
    background:#fafafa!important; border-radius:12px!important;
}
</style>
""", unsafe_allow_html=True)

# ── inline-style helpers ──────────────────────────────────────────
C = {
    "title":   "font-family:'DM Serif Display',serif;font-size:2.6rem;line-height:1.2;color:#eceaff;margin-bottom:.5rem",
    "subtext": "font-size:15px;color:#ccc;line-height:1.7",
    "infobox": "background:#f8f7ff;border-left:3px solid #7F77DD;border-radius:8px;padding:14px 16px;font-size:13px;color:#333;line-height:1.7;margin:1.5rem 0",
    "divider": "height:1px;background:rgba(255,255,255,0.12);margin:2rem 0",
    "sechead": "font-family:'DM Serif Display',serif;font-size:1.3rem;color:#eceaff;margin-bottom:1rem",
    "q_num":   "font-size:13px;color:#aaa;margin-bottom:2px",
    "q_text":  "font-family:'DM Serif Display',serif;font-size:1.45rem;line-height:1.5;color:#f0eeff;margin:.5rem 0 1.8rem",
    "scale":   "display:flex;justify-content:space-between;font-size:12px;color:#bbb;margin-bottom:12px;padding:0 4px",
    "progcap": "display:flex;justify-content:space-between;font-size:13px;color:#aaa;margin:-8px 0 20px;padding:0 4px",
    "pill_row":"display:flex;flex-wrap:wrap;gap:8px;margin:1rem 0 .5rem",
}

BADGE = {
    "EXT": "background:#EEEDFE;color:#3C3489;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600",
    "EST": "background:#FAECE7;color:#7A2810;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600",
    "AGR": "background:#E1F5EE;color:#085041;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600",
    "CSN": "background:#E6F1FB;color:#0C447C;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600",
    "OPN": "background:#FAEEDA;color:#633806;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600",
}

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# 43 features — ลำดับตรงกับที่ model เทรน (จาก error log)
MODEL_FEATURES = [
    'EXT1','EXT2','EXT3','EXT5','EXT6','EXT7','EXT8','EXT9','EXT10',
    'EST1','EST2','EST3','EST4','EST5','EST6','EST8','EST9','EST10',
    'AGR1','AGR2','AGR3','AGR4','AGR6','AGR7','AGR8','AGR10',
    'CSN1','CSN2','CSN3','CSN4','CSN5','CSN7','CSN8','CSN9','CSN10',
    'OPN1','OPN2','OPN4','OPN5','OPN6','OPN7','OPN8','OPN9',
]  # 43 features (50 - 7 targets)

# 7 targets ที่ถูก drop ออก (ไม่ถามในแบบทดสอบ — model จะทำนายแทน)
TARGET_COLS = {'CSN6','AGR9','AGR5','EXT4','EST7','OPN3','OPN10'}

# คำถาม 43 ข้อ = MODEL_FEATURES เท่านั้น (ไม่รวม target)
QUESTIONS = [
    # ── EXT (9 ข้อ — ตัด EXT4 ออก) ─────────────────────────────────
    {"id":"EXT1",  "trait":"EXT", "text":"ฉันเป็นจุดศูนย์กลางของงานปาร์ตี้"},
    {"id":"EXT2",  "trait":"EXT", "text":"ฉันไม่ค่อยพูดมาก",                     "reverse":True},
    {"id":"EXT3",  "trait":"EXT", "text":"ฉันรู้สึกสบายใจเมื่ออยู่กับคนอื่น"},
    {"id":"EXT5",  "trait":"EXT", "text":"ฉันเป็นคนริเริ่มสนทนา"},
    {"id":"EXT6",  "trait":"EXT", "text":"ฉันแทบไม่มีอะไรจะพูด",                 "reverse":True},
    {"id":"EXT7",  "trait":"EXT", "text":"ฉันพูดคุยกับคนหลายๆ คนในงานปาร์ตี้"},
    {"id":"EXT8",  "trait":"EXT", "text":"ฉันไม่อยากให้ใครสนใจตัวเอง",           "reverse":True},
    {"id":"EXT9",  "trait":"EXT", "text":"ฉันไม่รู้สึกอะไรถ้าต้องเป็นจุดสนใจ"},
    {"id":"EXT10", "trait":"EXT", "text":"ฉันเงียบเมื่ออยู่กับคนแปลกหน้า",       "reverse":True},
    # ── EST (9 ข้อ — ตัด EST7 ออก) ─────────────────────────────────
    {"id":"EST1",  "trait":"EST", "text":"ฉันรู้สึกเครียดได้ง่าย"},
    {"id":"EST2",  "trait":"EST", "text":"ฉันรู้สึกผ่อนคลายเกือบตลอดเวลา",       "reverse":True},
    {"id":"EST3",  "trait":"EST", "text":"ฉันมักกังวลเรื่องต่างๆ"},
    {"id":"EST4",  "trait":"EST", "text":"ฉันไม่ค่อยรู้สึกหดหู่",                 "reverse":True},
    {"id":"EST5",  "trait":"EST", "text":"ฉันถูกรบกวนอารมณ์ได้ง่าย"},
    {"id":"EST6",  "trait":"EST", "text":"ฉันหัวร้อนง่าย"},
    {"id":"EST8",  "trait":"EST", "text":"ฉันมีอารมณ์แปรปรวนบ่อยครั้ง"},
    {"id":"EST9",  "trait":"EST", "text":"ฉันหงุดหงิดง่าย"},
    {"id":"EST10", "trait":"EST", "text":"ฉันมักรู้สึกหดหู่บ่อยๆ"},
    # ── AGR (8 ข้อ — ตัด AGR5, AGR9 ออก) ──────────────────────────
    {"id":"AGR1",  "trait":"AGR", "text":"ฉันไม่ค่อยสนใจคนอื่น",                 "reverse":True},
    {"id":"AGR2",  "trait":"AGR", "text":"ฉันสนใจในตัวคน"},
    {"id":"AGR3",  "trait":"AGR", "text":"ฉันชอบพูดจาดูถูกคนอื่น",               "reverse":True},
    {"id":"AGR4",  "trait":"AGR", "text":"ฉันเข้าใจความรู้สึกของผู้อื่น"},
    {"id":"AGR6",  "trait":"AGR", "text":"ฉันมีจิตใจอ่อนโยน"},
    {"id":"AGR7",  "trait":"AGR", "text":"ฉันไม่ค่อยสนใจคนรอบข้าง",             "reverse":True},
    {"id":"AGR8",  "trait":"AGR", "text":"ฉันสละเวลาให้คนอื่นเสมอ"},
    {"id":"AGR10", "trait":"AGR", "text":"ฉันทำให้คนอื่นรู้สึกสบายใจ"},
    # ── CSN (9 ข้อ — ตัด CSN6 ออก) ─────────────────────────────────
    {"id":"CSN1",  "trait":"CSN", "text":"ฉันเตรียมพร้อมอยู่เสมอ"},
    {"id":"CSN2",  "trait":"CSN", "text":"ฉันมักทิ้งของไว้กระจัดกระจาย",         "reverse":True},
    {"id":"CSN3",  "trait":"CSN", "text":"ฉันใส่ใจในรายละเอียด"},
    {"id":"CSN4",  "trait":"CSN", "text":"ฉันมักทำให้สิ่งต่างๆ รกรุงรัง",        "reverse":True},
    {"id":"CSN5",  "trait":"CSN", "text":"ฉันทำงานบ้านเสร็จทันที"},
    {"id":"CSN7",  "trait":"CSN", "text":"ฉันชอบความเป็นระเบียบ"},
    {"id":"CSN8",  "trait":"CSN", "text":"ฉันมักหลีกเลี่ยงหน้าที่รับผิดชอบ",     "reverse":True},
    {"id":"CSN9",  "trait":"CSN", "text":"ฉันทำตามตารางเวลา"},
    {"id":"CSN10", "trait":"CSN", "text":"ฉันทำงานอย่างละเอียดรอบคอบ"},
    # ── OPN (8 ข้อ — ตัด OPN3, OPN10 ออก) ─────────────────────────
    {"id":"OPN1",  "trait":"OPN", "text":"ฉันมีคลังคำศัพท์ที่หลากหลาย"},
    {"id":"OPN2",  "trait":"OPN", "text":"ฉันมีความยากในการเข้าใจแนวคิดนามธรรม", "reverse":True},
    {"id":"OPN4",  "trait":"OPN", "text":"ฉันไม่สนใจแนวคิดนามธรรม",             "reverse":True},
    {"id":"OPN5",  "trait":"OPN", "text":"ฉันมีไอเดียที่ดีเยี่ยม"},
    {"id":"OPN6",  "trait":"OPN", "text":"ฉันไม่ค่อยมีจินตนาการ",                "reverse":True},
    {"id":"OPN7",  "trait":"OPN", "text":"ฉันเข้าใจสิ่งต่างๆ ได้รวดเร็ว"},
    {"id":"OPN8",  "trait":"OPN", "text":"ฉันชอบใช้คำที่ซับซ้อน"},
    {"id":"OPN9",  "trait":"OPN", "text":"ฉันใช้เวลาไตร่ตรองสิ่งต่างๆ"},
]
# ตรวจสอบ: len(QUESTIONS) ต้องเท่ากับ len(MODEL_FEATURES) = 43
assert len(QUESTIONS) == len(MODEL_FEATURES) == 43, \
    f"QUESTIONS={len(QUESTIONS)}, MODEL_FEATURES={len(MODEL_FEATURES)}"

REVERSE_IDS = {q["id"] for q in QUESTIONS if q.get("reverse")}

TRAIT_INFO = {
    "EXT": {"name":"Extraversion",      "th":"ความเปิดเผย",   "color":"#534AB7","bg":"#EEEDFE","tc":"#534AB7"},
    "EST": {"name":"Neuroticism",       "th":"ความวิตกกังวล","color":"#993C1D","bg":"#FAECE7","tc":"#993C1D"},
    "AGR": {"name":"Agreeableness",     "th":"ความเป็นมิตร",  "color":"#0F6E56","bg":"#E1F5EE","tc":"#0F6E56"},
    "CSN": {"name":"Conscientiousness", "th":"ความมีระเบียบ", "color":"#185FA5","bg":"#E6F1FB","tc":"#185FA5"},
    "OPN": {"name":"Openness",          "th":"ความเปิดรับ",   "color":"#854F0B","bg":"#FAEEDA","tc":"#854F0B"},
}

BEHAVIORS = [
    {"col":"CSN6",  "label":"ลืมวางของกลับที่เดิม",  "desc":"มีแนวโน้มลืมวางของในที่เดิมบ่อยแค่ไหน",           "trait":"CSN","low":"ไม่ค่อยลืม","high":"ลืมบ่อยมาก"},
    {"col":"AGR9",  "label":"รับรู้อารมณ์คนอื่น",    "desc":"สามารถรู้สึกและเข้าใจอารมณ์คนรอบข้างได้ดี",       "trait":"AGR","low":"น้อย",      "high":"มากมาย"},
    {"col":"AGR5",  "label":"สนใจปัญหาคนอื่น",       "desc":"ใส่ใจและให้ความสำคัญกับปัญหาของผู้อื่น",          "trait":"AGR","low":"ไม่สนใจ",   "high":"สนใจมาก"},
    {"col":"EXT4",  "label":"ชอบอยู่เบื้องหลัง",     "desc":"ชอบอยู่ในพื้นหลัง ไม่ต้องการเป็นจุดสนใจ",         "trait":"EXT","low":"ชอบเด่น",   "high":"ชอบซ่อนตัว"},
    {"col":"EST7",  "label":"อารมณ์แปรปรวน",         "desc":"อารมณ์เปลี่ยนแปลงได้บ่อยและรวดเร็วแค่ไหน",        "trait":"EST","low":"มั่นคง",    "high":"แปรปรวนมาก"},
    {"col":"OPN3",  "label":"จินตนาการสูง",           "desc":"มีจินตนาการลึกซึ้งและมีภาพในหัวชัดเจน",           "trait":"OPN","low":"น้อย",      "high":"สูงมาก"},
    {"col":"OPN10", "label":"มีไอเดียมาก",            "desc":"เต็มไปด้วยความคิดและแนวคิดใหม่ๆ อยู่เสมอ",    "trait":"OPN","low":"น้อย",      "high":"มาก"},
]

# ─────────────────────────────────────────────
# Load models  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_model(col):
    FILE_IDS = {
        "AGR5": "12PMvfPLH_yknObS_HQtBlYKDm5fAy6MH",
        "AGR9": "157oXBN-bF6DVLodI4ARFavXXt1odNZED",
        "CSN6": "116n0v9KM8LJXaC5twYjp_OvjNJHOd0c-",
        "EST7": "1n57mJYEGB8Z60XhRIphL9rHT3vU6Y1BN",
        "EXT4": "19Wsxb5DISjQw2aCRI60_dkragbg2uzw3",
        "OPN3": "1RN53hZR86pXJ8VyRlHFqmIhkYfr0EZB7",
        "OPN10": "1VdyM_u_8aXNhKtCGiYWxjEApvEI8sSMk",
    }

    os.makedirs("models", exist_ok=True)
    path = f"models/xgb_{col}.pkl"

    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={FILE_IDS[col]}"
        gdown.download(url, path, quiet=False)

    with open(path, "rb") as f:
        return pickle.load(f)
# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    for k, v in [("page","welcome"),("q_idx",0),("answers",{})]:
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────
def apply_reverse(qid: str, raw: int) -> int:
    return (6 - raw) if qid in REVERSE_IDS else raw

def build_feature_vector(answers: dict) -> pd.DataFrame:
    """
    answers: {qid: raw_1_to_5}
    Returns DataFrame with columns = MODEL_FEATURES, values already reverse-scored.
    """
    processed = {qid: apply_reverse(qid, val) for qid, val in answers.items()}
    row = {feat: processed.get(feat, 3) for feat in MODEL_FEATURES}
    return pd.DataFrame([row], columns=MODEL_FEATURES)

def compute_trait_means(answers: dict) -> dict:
    trait_vals = {t: [] for t in TRAIT_INFO}
    for q in QUESTIONS:
        qid = q["id"]
        if qid in answers:
            trait_vals[q["trait"]].append(apply_reverse(qid, answers[qid]))
    return {t: (float(np.mean(v)) if v else 3.0) for t, v in trait_vals.items()}

def predict_behaviors(answers: dict) -> dict:
    X = build_feature_vector(answers)
    preds = {}

    for col in ["AGR5","AGR9","CSN6","EST7","EXT4","OPN3","OPN10"]:
        model = get_model(col)   # 🔥 โหลดทีละตัว
        val = float(model.predict(X)[0])
        preds[col] = float(np.clip(val, 1.0, 5.0))

    return preds

def score_label(s: float) -> str:
    if s < 2.0:  return "ต่ำมาก"
    if s < 2.75: return "ต่ำ"
    if s < 3.5:  return "ปานกลาง"
    if s < 4.25: return "สูง"
    return "สูงมาก"

def score_bar(score: float, color: str, h: int = 8) -> str:
    pct = (score - 1) / 4 * 100
    return (
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
    )

# ─────────────────────────────────────────────
# PAGE: Welcome
# ─────────────────────────────────────────────
if st.session_state.page == "welcome":

    st.markdown(f'<div style="{C["title"]}">Big Five<br>Personality Test</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="{C["subtext"]}">ค้นพบบุคลิกภาพ 5 ด้านของคุณ และทำนายพฤติกรรมเฉพาะตัว 7 ประการ '
        'ด้วย XGBoost Model ที่เทรนจากผู้ทดสอบกว่า 1 ล้านคน</p>',
        unsafe_allow_html=True,
    )

    pills_html = f'<div style="{C["pill_row"]}">'
    for trait, info in TRAIT_INFO.items():
        pills_html += (
            f'<span style="display:inline-block;{BADGE[trait]};margin-bottom:6px">'
            f'{info["name"]} <span style="opacity:0.75;font-weight:400">— {info["th"]}</span>'
            f'</span>'
        )
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown(
        f'<div style="{C["infobox"]}">'
        '<b>วิธีตอบ:</b> กด 1–5 เพื่อเลือกคำตอบ โดย 1 = ไม่เห็นด้วยอย่างยิ่ง '
        'และ 5 = เห็นด้วยอย่างยิ่ง หลังกดจะไปข้อถัดไปทันที<br>'
        '<span style="color:#555">✦ จำนวนคำถาม: 43 ข้อ &nbsp;|&nbsp; เวลาประมาณ 5–8 นาที</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("เริ่มทำแบบทดสอบ →", use_container_width=True, type="primary"):
        st.session_state.page    = "quiz"
        st.session_state.q_idx   = 0
        st.session_state.answers = {}
        st.rerun()

# ─────────────────────────────────────────────
# PAGE: Quiz
# ─────────────────────────────────────────────
elif st.session_state.page == "quiz":

    total = len(QUESTIONS)
    idx   = st.session_state.q_idx
    q     = QUESTIONS[idx]
    info  = TRAIT_INFO[q["trait"]]

    st.progress(idx / total)
    st.markdown(
        f'<div style="{C["progcap"]}">'
        f'<span>ข้อ {idx+1} / {total}</span>'
        f'<span style="color:#7F77DD;font-weight:600">{int(idx/total*100)}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<span style="display:inline-block;{BADGE[q["trait"]]};margin-bottom:10px">'
        f'{info["name"]} — {info["th"]}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div style="{C["q_text"]}">{q["text"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="{C["scale"]}">'
        '<span>ไม่เห็นด้วยอย่างยิ่ง</span>'
        '<span>กลางๆ</span>'
        '<span>เห็นด้วยอย่างยิ่ง</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    btn_cols = st.columns(5)
    for i, col in enumerate(btn_cols):
        if col.button(str(i + 1), key=f"ans_{idx}_{i}", use_container_width=True):
            st.session_state.answers[q["id"]] = i + 1
            if idx + 1 < total:
                st.session_state.q_idx += 1
            else:
                st.session_state.page = "result"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if idx > 0:
        if st.button("← ย้อนกลับ", key="back", type="secondary"):
            st.session_state.q_idx -= 1
            st.rerun()

# ─────────────────────────────────────────────
# PAGE: Result
# ─────────────────────────────────────────────
elif st.session_state.page == "result":

    answers   = st.session_state.answers
    traits    = compute_trait_means(answers)
    behaviors = predict_behaviors(answers)

    st.markdown(f'<div style="{C["title"]}">ผลของคุณ ✦</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="{C["subtext"]}">วิเคราะห์ด้วย XGBoost Model '
        'ที่เทรนจากชุดข้อมูล IPIP Big Five ของผู้ทดสอบกว่า 1 ล้านคน (2016–2018)</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="{C["divider"]}"></div>', unsafe_allow_html=True)

    # ── Big Five ──────────────────────────────────────────────────
    st.markdown(f'<div style="{C["sechead"]}">บุคลิกภาพ 5 ด้าน</div>', unsafe_allow_html=True)
    for trait, info in TRAIT_INFO.items():
        score = traits[trait]
        lbl   = score_label(score)
        pct   = (score - 1) / 4 * 100
        st.markdown(
            f'<div style="margin-bottom:16px">'
            f'  <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:5px">'
            f'    <span><b style="color:{info["tc"]}">{info["name"]}</b>'
            f'    &nbsp;<span style="color:#aaa;font-size:12px;font-weight:400">{info["th"]}</span></span>'
            f'    <span style="color:#ddd">{score:.2f} / 5 &nbsp;'
            f'      <span style="font-size:12px;color:#999">({lbl})</span></span>'
            f'  </div>'
            f'  <div style="height:8px;background:rgba(255,255,255,0.12);border-radius:99px;overflow:hidden">'
            f'    <div style="width:{pct:.1f}%;height:100%;background:{info["color"]};border-radius:99px"></div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div style="{C["divider"]}"></div>', unsafe_allow_html=True)

    # ── Predicted behaviors ───────────────────────────────────────
    st.markdown(f'<div style="{C["sechead"]}">การทำนายพฤติกรรม 7 ประการ</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#bbb;margin-bottom:1rem;line-height:1.75">'
        'ทำนายจากโมเดล XGBoost 7 ตัวแยกกัน โดยใช้คะแนน 43 ข้อ </p>',
        unsafe_allow_html=True,
    )

    for b in BEHAVIORS:
        score = behaviors[b["col"]]
        lbl   = score_label(score)
        info  = TRAIT_INFO[b["trait"]]
        pct   = (score - 1) / 4 * 100
        badge_s = BADGE[b["trait"]]

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);'
            f'border-radius:16px;padding:18px 20px;margin-bottom:12px">'
            f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
            f'    <div style="flex:1">'
            f'      <div style="font-weight:600;font-size:15px;color:#f0eeff;margin-bottom:4px">'
            f'        {b["label"]}'
            f'        <span style="display:inline-block;{badge_s};margin-left:8px;vertical-align:middle;font-size:10px">'
            f'          {info["name"]}</span>'
            f'      </div>'
            f'      <div style="font-size:13px;color:#aaa;margin-bottom:12px">{b["desc"]}</div>'
            f'    </div>'
            f'    <div style="font-family:DM Serif Display,serif;font-size:28px;color:{info["color"]};flex-shrink:0">'
            f'      {score:.1f}</div>'
            f'  </div>'
            f'  <div style="display:flex;justify-content:space-between;font-size:11px;color:#888;margin-bottom:6px">'
            f'    <span>{b["low"]}</span><span>{b["high"]}</span></div>'
            f'  <div style="height:6px;background:rgba(255,255,255,0.1);border-radius:99px;overflow:hidden">'
            f'    <div style="width:{pct:.1f}%;height:100%;background:{info["color"]};border-radius:99px"></div>'
            f'  </div>'
            f'  <div style="text-align:right;font-size:12px;color:{info["color"]};font-weight:700;margin-top:6px">'
            f'    {lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div style="{C["divider"]}"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(127,119,221,0.12);border-left:3px solid #7F77DD;'
        'border-radius:8px;padding:14px 16px;font-size:12px;color:#aaa;line-height:1.8;margin-bottom:1.5rem">'
        'ผลนี้เป็นการประมาณการจาก Machine Learning Model '
        'ไม่ใช่การวินิจฉัยทางจิตวิทยาหรือสุขภาพจิต '
        'คะแนนอยู่ในมาตรวัด 1–5 ● '
        'ข้อมูลต้นฉบับ: IPIP Big Five Personality Test (2016–2018)'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("ทำแบบทดสอบอีกครั้ง", use_container_width=True, type="primary"):
        st.session_state.page    = "welcome"
        st.session_state.q_idx   = 0
        st.session_state.answers = {}
        st.rerun()
