# app.py — DietPost AI  |  Tek dosya, Streamlit Cloud uyumlu
# ─────────────────────────────────────────────────────────────────────────────
import os, json, re, textwrap
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — SABİTLER & TABLOLAR
# ═════════════════════════════════════════════════════════════════════════════

FREQ_SCORE: dict = {
    "never": 0, "less often": 0.5, "once a month": 1,
    "few times a week": 3, "often": 4, "once a day": 7,
    "several times a day": 14, "in every meal": 21,
    "very frequently": 14, "with every meal": 21,
}

WATER_ML: dict = {
    "less than 3 cups": 500, "4-6 cups": 1250,
    "7-10 cups": 2125, "11-14 cups": 3125, "more than 15 cups": 4000,
}

AGE_MIDPOINT: dict = {
    "under 18": 16, "18-24": 21, "25-34": 29,
    "35-44": 39, "45-54": 49, "55-64": 59,
    "65-74": 69, "above 65": 70,
}

SKIP_SCORE: dict = {
    "never": 0, "rarely": 1, "sometimes": 2, "often": 3, "very frequently": 4,
}

DIET_SHORT: dict = {
    "non-vegetarian": "Omnivore", "vegetarian": "Vejetaryen",
    "eggetarian": "Eggetaryen", "pollotarian": "Pollotaryen",
    "pescatarian": "Pescataryen", "pollo-pescetarian": "Pollo-Pesc.",
}

FREQ_COLS = ["sweet","salty","fruit","veg","fried","meat","seafood",
             "tea","coffee","soda","juice","dairy","alcohol"]

RISK_META: dict = {
    "insulin_resistance": {
        "label": "İnsülin Direnci", "icon": "🩺", "color": "#EF4444",
        "diet_tip": "Düşük GI diyet · Şeker kısıtlaması · Ara öğün düzeni",
        "post_topic": "İnsülin direncini kontrol eden beslenme alışkanlıkları",
        "cta": "Diyetisyeninizle insülin direnci beslenme planı oluşturun!",
    },
    "weight_gain": {
        "label": "Kilo Alım Riski", "icon": "⚖️", "color": "#F97316",
        "diet_tip": "Kalori dengesi · İşlenmiş gıda kısıtlaması · Porsiyon kontrolü",
        "post_topic": "Kilo kontrolü için kaçınılması gereken 5 alışkanlık",
        "cta": "Sağlıklı kilo yönetimi için beslenme danışmanlığı alın!",
    },
    "nutrient_deficiency": {
        "label": "Besin Eksikliği", "icon": "🥦", "color": "#10B981",
        "diet_tip": "Günde 5 porsiyon meyve-sebze · Çeşitlilik · Renk çeşitliliği",
        "post_topic": "Günlük 5 porsiyon meyve-sebze neden bu kadar önemli?",
        "cta": "Beslenme planınıza renk katın, meyve-sebze eksikliğini giderin!",
    },
    "protein_deficiency": {
        "label": "Protein Yetersizliği", "icon": "💪", "color": "#8B5CF6",
        "diet_tip": "Baklagiller · Tofu · Quinoa · Süt ürünleri · Yumurta",
        "post_topic": "Vejetaryenler için en iyi bitkisel protein kaynakları",
        "cta": "Bitkisel protein kaynaklarını keşfedin!",
    },
    "dehydration": {
        "label": "Dehidrasyon Riski", "icon": "💧", "color": "#3B82F6",
        "diet_tip": "Günde 8-10 bardak su · Su içerikli meyve-sebze",
        "post_topic": "Günde kaç bardak su içmelisiniz? Bilimsel yanıt",
        "cta": "Yeterli su içmek metabolizmanızı hızlandırır!",
    },
    "meal_skipping": {
        "label": "Öğün Atlama", "icon": "⏰", "color": "#EAB308",
        "diet_tip": "Düzenli öğün programı · Kahvaltı önemi · Ara öğün planlaması",
        "post_topic": "Öğün atlamanın vücudunuza gerçek etkisi",
        "cta": "Düzenli beslenme programı oluşturun!",
    },
    "excess_sugar_drinks": {
        "label": "Aşırı Şekerli İçecek", "icon": "🥤", "color": "#EC4899",
        "diet_tip": "Şekerli içecek yerine su/bitki çayı · Etiket okuma",
        "post_topic": "Gazlı içeceklerin sağlığa gizli zararları",
        "cta": "Şekerli içecekleri hayatınızdan çıkarın!",
    },
    "healthy": {
        "label": "Genel Sağlık", "icon": "✅", "color": "#10B981",
        "diet_tip": "Sağlıklı alışkanlıkları sürdürme · Çeşitli beslenme",
        "post_topic": "Sağlıklı beslenme alışkanlıklarınızı koruyun",
        "cta": "Sağlıklı yaşam yolculuğunuzda yanınızdayız!",
    },
}

PLATFORMS       = ["Instagram", "TikTok", "Facebook", "Twitter/X", "LinkedIn"]
CHAR_LIMIT      = {"Instagram": 2200, "TikTok": 150, "Facebook": 500,
                   "Twitter/X": 280, "LinkedIn": 700}
POST_TYPES      = {"carousel": "📊 Carousel", "reel": "🎬 Reel",
                   "story": "📱 Story", "infographic": "📈 İnfografik", "post": "🖼️ Post"}
BEST_TIMES      = {"Instagram": "Salı/Perşembe 11:00–13:00 veya 19:00–21:00",
                   "TikTok": "Akşam 19:00–22:00 (hafta içi)",
                   "Facebook": "Çarşamba 13:00–15:00",
                   "Twitter/X": "Sabah 08:00–10:00 veya öğlen 12:00",
                   "LinkedIn": "Salı–Perşembe 09:00–11:00"}

# ═════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — VERİ TEMİZLEME
# ═════════════════════════════════════════════════════════════════════════════

def _build_rename(cols):
    m = {}
    for i, c in enumerate(cols):
        cl = c.lower().strip()
        if   i == 0:                                  m[c] = "age_raw"
        elif i == 1:                                  m[c] = "gender"
        elif "meals" in cl and "day" in cl:           m[c] = "meals_day"
        elif "best describe" in cl:                   m[c] = "diet_type"
        elif "skip meals" in cl:                      m[c] = "skip_meals"
        elif "hunger" in cl:                          m[c] = "hunger"
        elif "consult" in cl:                         m[c] = "consult_diet"
        elif "cook" in cl:                            m[c] = "cook_own"
        elif "main meal" in cl:                       m[c] = "main_meal"
        elif "prepared" in cl or ("consist" in cl):   m[c] = "food_prep"
        elif "order-in" in cl or "go out to eat" in cl: m[c] = "eat_out"
        elif "allergic" in cl:                        m[c] = "allergies"
        elif "sweet" in cl:                           m[c] = "sweet"
        elif "salty" in cl:                           m[c] = "salty"
        elif "fresh fruit" in cl:                     m[c] = "fruit"
        elif "fresh veg" in cl:                       m[c] = "veg"
        elif "oily" in cl or "fried" in cl:           m[c] = "fried"
        elif "[meat]" in cl or ("food categor" in cl and "meat" in cl): m[c] = "meat"
        elif "seafood" in cl:                         m[c] = "seafood"
        elif "[tea]" in cl:                           m[c] = "tea"
        elif "[coffee]" in cl:                        m[c] = "coffee"
        elif "aerated" in cl or "soft drink" in cl:   m[c] = "soda"
        elif "fruit juice" in cl:                     m[c] = "juice"
        elif "dairy" in cl or ("milk" in cl and "beverage" in cl): m[c] = "dairy"
        elif "alcoholic" in cl:                       m[c] = "alcohol"
        elif "water" in cl:                           m[c] = "water_raw"
    return m


def clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # İlk 26 sütunu al (boş son sütunları at)
    df = df.iloc[:, :26]
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns=_build_rename(df.columns.tolist()))

    # Yaş → sayı
    df["age"] = df["age_raw"].str.lower().str.strip().map(AGE_MIDPOINT)

    # Cinsiyet normalize
    df["gender"] = df["gender"].str.strip().str.capitalize()

    # Frekans → skor
    for c in FREQ_COLS:
        if c in df.columns:
            df[f"{c}_s"] = df[c].str.lower().str.strip().map(FREQ_SCORE)

    # Su → ml
    if "water_raw" in df.columns:
        df["water_ml"] = df["water_raw"].str.lower().str.strip().map(WATER_ML)
    else:
        df["water_ml"] = np.nan

    # Öğün atlama skoru
    if "skip_meals" in df.columns:
        df["skip_score"] = df["skip_meals"].str.lower().str.strip().map(SKIP_SCORE)
    else:
        df["skip_score"] = 0

    # Diyet tipi kısaltma
    def _short(val):
        if pd.isna(val): return "Bilinmiyor"
        vl = str(val).lower()
        for k, v in DIET_SHORT.items():
            if k in vl: return v
        return str(val).split("(")[0].strip()

    if "diet_type" in df.columns:
        df["diet_short"] = df["diet_type"].apply(_short)
    else:
        df["diet_short"] = "Bilinmiyor"

    # Sağlık skoru (0-100)
    def _health(r):
        s = 50.0
        s += (r.get("fruit_s") or 0) * 0.5
        s += (r.get("veg_s")   or 0) * 0.5
        s -= (r.get("sweet_s") or 0) * 0.4
        s -= (r.get("fried_s") or 0) * 0.5
        s -= (r.get("soda_s")  or 0) * 0.6
        s += ((r.get("water_ml") or 0) - 1500) / 500 * 2
        s -= (r.get("skip_score") or 0) * 2
        return round(max(0.0, min(100.0, s)), 1)

    df["health_score"] = df.apply(lambda r: _health(r.to_dict()), axis=1)

    # Risk sınıflandırması
    def _risks(r):
        age  = r.get("age") or 0
        sw   = r.get("sweet_s") or 0
        fr   = r.get("fried_s") or 0
        vg   = r.get("veg_s")   or 0
        ft   = r.get("fruit_s") or 0
        wt   = r.get("water_ml") or 0
        sk   = r.get("skip_score") or 0
        meat = r.get("meat_s")  or 0
        soda = r.get("soda_s")  or 0

        risks = []
        if age >= 35 and sw >= 7 and fr >= 3:
            risks.append("insulin_resistance")
        if sw >= 7 or fr >= 7 or (sw >= 4 and fr >= 4):
            risks.append("weight_gain")
        if ft < 1 and vg < 3:
            risks.append("nutrient_deficiency")
        if meat < 1 and r.get("diet_short") in ["Vejetaryen", "Eggetaryen"]:
            risks.append("protein_deficiency")
        if wt < 1250:
            risks.append("dehydration")
        if sk >= 3:
            risks.append("meal_skipping")
        if soda >= 7:
            risks.append("excess_sugar_drinks")
        return risks if risks else ["healthy"]

    df["risks"]      = df.apply(lambda r: _risks(r.to_dict()), axis=1)
    df["risk_count"] = df["risks"].apply(len)
    df["top_risk"]   = df["risks"].apply(lambda rs: rs[0])
    return df


def df_to_excel(df: pd.DataFrame, sheet: str = "Veri") -> bytes:
    buf = BytesIO()
    export_cols = [c for c in ["age_raw","age","gender","diet_short","meals_day",
                                "skip_meals","consult_diet","water_ml","health_score",
                                "risks","risk_count","top_risk"] if c in df.columns]
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df[export_cols].to_excel(writer, index=False, sheet_name=sheet)
        wb = writer.book
        ws = writer.sheets[sheet]
        hf = wb.add_format({"bold": True, "bg_color": "#1E40AF",
                             "font_color": "white", "border": 1, "align": "center"})
        df2 = wb.add_format({"border": 1, "align": "center"})
        for ci, col in enumerate(export_cols):
            ws.write(0, ci, col, hf)
            w = max(df[col].astype(str).str.len().max(), len(col)) + 2
            ws.set_column(ci, ci, min(w, 35), df2)
        ws.freeze_panes(1, 0)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — ANALİZ FONKSİYONLARI
# ═════════════════════════════════════════════════════════════════════════════

def get_kpis(df: pd.DataFrame) -> dict:
    all_risks = [r for rs in df["risks"] for r in rs]
    top_risk  = Counter(all_risks).most_common(1)[0][0] if all_risks else "healthy"
    return {
        "n":          len(df),
        "avg_health": round(df["health_score"].mean(), 1),
        "pct_multi_risk": round((df["risk_count"] > 1).mean() * 100, 1),
        "pct_no_consult": round(
            (df["consult_diet"].str.lower() == "never").mean() * 100, 1
        ) if "consult_diet" in df.columns else 0,
        "top_risk": top_risk,
    }


def top_risks_by_age(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for age_grp, grp in df.groupby("age_raw"):
        all_r = [r for rs in grp["risks"] for r in rs]
        total = len(grp)
        for rank, (risk, cnt) in enumerate(Counter(all_r).most_common(3), 1):
            meta = RISK_META.get(risk, {})
            rows.append({
                "age_group":  age_grp,
                "rank":       rank,
                "risk_key":   risk,
                "risk_label": meta.get("label", risk),
                "risk_icon":  meta.get("icon", ""),
                "risk_color": meta.get("color", "#888"),
                "count":      cnt,
                "total":      total,
                "pct":        round(cnt / total * 100, 1),
                "diet_tip":   meta.get("diet_tip", ""),
                "post_topic": meta.get("post_topic", ""),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — POST KARAR MOTORU
# ═════════════════════════════════════════════════════════════════════════════

def _hashtags(risk_key: str, age_group: str, platform: str) -> list:
    base = {
        "insulin_resistance":  ["#insulindirenci","#glisemikindeks","#kanşekeri"],
        "weight_gain":         ["#kilokontrolü","#sağlıklıkilo","#beslenmetavsiyesi"],
        "nutrient_deficiency": ["#meyvesebze","#besineksikliği","#sağlıklıbeslenme"],
        "protein_deficiency":  ["#bitkiselprotein","#vejeteryan","#protein"],
        "dehydration":         ["#suiçin","#hidrasyon","#günlüksu"],
        "meal_skipping":       ["#öğünatlamayın","#düzenlibeslenme","#kahvaltı"],
        "excess_sugar_drinks": ["#şekersiz","#sağlıklıiçecek","#gazlıiçecek"],
        "healthy":             ["#sağlıklıyaşam","#beslenme","#wellness"],
    }.get(risk_key, [])
    age_tags = {
        "under 18": ["#gençbeslenme","#öğrencisağlığı"],
        "18-24":    ["#gençler","#üniversitebeslenme"],
        "35-44":    ["#35yas","#metabolizma"],
        "45-54":    ["#45yas","#sağlıklıyaşlanma"],
        "above 65": ["#65yas","#yaşlısağlığı"],
    }.get(age_group.lower(), [])
    tags = ["#diyetisyen", "#beslenme"] + base + age_tags
    if platform == "Instagram":
        tags.append("#healthylifestyle")
    return tags[:8]


def _visual_tip(risk_key: str, post_type: str) -> str:
    base = {
        "insulin_resistance":  "Yeşil tonlarda GI karşılaştırma tablosu",
        "weight_gain":         "Önce-sonra porsiyon karşılaştırması",
        "nutrient_deficiency": "Gökkuşağı tabağı — renkli meyve-sebze düzenlemesi",
        "protein_deficiency":  "Bitkisel protein kaynakları kolajı",
        "dehydration":         "Günlük su takip çizelgesi animasyonu",
        "meal_skipping":       "Saat kadranı üzerinde öğün zamanları",
        "excess_sugar_drinks": "'Bunun yerine bunu iç' karşılaştırma formatı",
        "healthy":             "Sağlıklı tabak düzenlemesi, pozitif renk paleti",
    }.get(risk_key, "Temiz, minimal tasarım")
    suffix = {"reel": " — 15sn montaj", "story": " — Anket kutusu ekle",
              "carousel": " — Her slayt tek madde"}.get(post_type, "")
    return base + suffix


def decide_post(age_group, risk_key, platform, post_type,
                patient_count, avg_health) -> dict:
    meta    = RISK_META.get(risk_key, RISK_META["healthy"])
    limit   = CHAR_LIMIT.get(platform, 500)
    age_map = {"under 18": "18 Yaş Altı", "18-24": "18-24 Yaş",
               "35-44": "35-44 Yaş", "45-54": "45-54 Yaş", "above 65": "65+ Yaş"}
    age_lbl = age_map.get(age_group.lower(), age_group)

    title   = f"{meta['icon']} {age_lbl} İçin: {meta['post_topic']}"
    caption = (
        f"{age_lbl} grubundaki bireyler için hazırladığımız bu içerikte "
        f"'{meta['label'].lower()}' konusunu ele alıyoruz.\n\n"
        f"✅ Uzman Tavsiyesi: {meta['diet_tip']}\n\n"
        f"{meta['cta']}"
    )
    if len(caption) > limit:
        caption = caption[:limit - 3] + "..."

    return {
        "title":      title,
        "caption":    caption,
        "hashtags":   _hashtags(risk_key, age_group, platform),
        "best_time":  BEST_TIMES.get(platform, "Haftanın ortası, öğlen saatleri"),
        "visual_tip": _visual_tip(risk_key, post_type),
        "cta":        meta["cta"],
        "diet_tip":   meta["diet_tip"],
        "why":        (f"{age_lbl} grubunda {meta['label'].lower()} riski öne çıkıyor "
                       f"({patient_count} kişi, ort. sağlık skoru {avg_health}/100)."),
        "source":     "template",
    }


def generate_ai_post(seg: dict, platform: str, post_type: str) -> dict | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        meta   = RISK_META.get(seg["risk_key"], {})
        prompt = textwrap.dedent(f"""
        Sen uzman bir diyetisyen sosyal medya danışmanısın.
        {platform} için {post_type} formatında Türkçe içerik üret.

        SEGMENT: {seg['age_group']} · {meta.get('label','')} · {seg['patient_count']} kişi
        DİYET İPUCU: {meta.get('diet_tip','')}
        SAĞLIK SKORU: {seg['avg_health']}/100

        SADECE JSON döndür:
        {{"title":"...","caption":"...","hashtags":["...","...","...","...","..."],
          "best_time":"...","visual_tip":"...","why":"..."}}
        """).strip()

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = re.sub(r"```json|```", "", msg.content[0].text).strip()
        result = json.loads(raw)
        result["source"] = "ai"
        return result
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — STREAMLIT ARAYÜZÜ
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DietPost AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebarNav"] { display: none; }

div[data-testid="metric-container"] {
    background:#1E293B; border:1px solid #334155;
    border-radius:12px; padding:1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color:#94A3B8 !important; font-size:0.74rem !important;
    text-transform:uppercase; letter-spacing:.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:1.65rem !important; font-weight:700 !important; color:#F1F5F9 !important;
}
.post-card {
    background:#1E293B; border:1px solid #334155; border-top:3px solid;
    border-radius:12px; padding:1.4rem 1.6rem; margin-bottom:1rem;
}
.post-title   { font-size:.98rem; font-weight:600; color:#F1F5F9; margin-bottom:7px; }
.post-caption { color:#CBD5E1; font-size:.86rem; line-height:1.65; margin-bottom:9px; }
.hashtag { display:inline-block; background:#1D4ED8; color:#BFDBFE;
           padding:2px 8px; border-radius:20px; font-size:.71rem; margin:2px; }
.pill    { display:inline-block; background:#0F172A; border:1px solid #334155;
           color:#94A3B8; padding:2px 9px; border-radius:20px; font-size:.71rem; margin:2px; }
.risk-row {
    background:#1E293B; border:1px solid #334155; border-radius:10px;
    padding:.9rem 1.1rem; margin-bottom:.5rem;
}
.step-h {
    background:linear-gradient(90deg,#1E293B,#0F172A);
    border-left:4px solid #2563EB; border-radius:0 8px 8px 0;
    padding:.55rem .9rem; margin:1.1rem 0 .7rem;
    font-weight:600; color:#F1F5F9; font-size:.95rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
        <div style='font-size:2.4rem'>🥗</div>
        <div style='font-size:1.15rem;font-weight:700;color:#F1F5F9'>DietPost AI</div>
        <div style='font-size:.7rem;color:#64748B'>CSV → Analiz → Sosyal Medya</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Veri")
    use_demo = st.toggle("Demo veri kullan", value=True)
    uploaded = None
    if not use_demo:
        uploaded = st.file_uploader("CSV / Excel yükle", type=["csv","xlsx","xls"])

    st.markdown("---")
    st.markdown("### 🔍 Filtreler")
    _age_ph    = st.empty()
    _gender_ph = st.empty()
    _diet_ph   = st.empty()

    st.markdown("---")
    st.markdown("### 📱 Post Ayarları")
    platform  = st.selectbox("Platform", PLATFORMS)
    post_type = st.selectbox("Format", list(POST_TYPES.keys()),
                             format_func=lambda x: POST_TYPES[x])

    st.markdown("---")
    st.markdown("### 🤖 AI (opsiyonel)")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="Olmadan şablon tabanlı içerik üretilir")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.success("✓ AI aktif")

# ── Veri yükleme ──────────────────────────────────────────────────────────────
DEMO_CSV = """Age,Gender,How many meals do you have a day? (number of regular occasions in a day when a significant and reasonably filling amount of food is eaten),What would best describe your diet:,Choose all that apply: [I skip meals],Choose all that apply: [I experience feelings of hunger during the day],Choose all that apply: [I consult a nutritionist/dietician],Choose all that apply: [I cook my own meals],What would you consider to be the main meal of YOUR day?,What does your diet mostly consist of and how is it prepared?,How many times a week do you order-in or go out to eat?,Are you allergic to any of the following? (Tick all that apply),What is your weekly food intake frequency of the following food categories: [Sweet foods],What is your weekly food intake frequency of the following food categories: [Salty foods],What is your weekly food intake frequency of the following food categories: [Fresh fruit],What is your weekly food intake frequency of the following food categories: [Fresh vegetables],"What is your weekly food intake frequency of the following food categories: [Oily, fried foods]",What is your weekly food intake frequency of the following food categories: [Meat],What is your weekly food intake frequency of the following food categories: [Seafood ],How frequently do you consume these beverages [Tea],How frequently do you consume these beverages [Coffee],How frequently do you consume these beverages [Aerated (Soft) Drinks],How frequently do you consume these beverages [Fruit Juices (Fresh/Packaged)],"How frequently do you consume these beverages [Dairy Beverages (Milk, Milkshakes, Smoothies, Buttermilk, etc)]",How frequently do you consume these beverages [Alcoholic Beverages],"What is your water consumption like (in a day, 1 cup=250ml approx)"
18-24,Male,3,Non-Vegetarian,Often,Often,Never,Sometimes,Lunch,Freshly home-cooked produce,3,I do not have any allergies,Once a day,Once a day,Few times a week,Once a day,Few times a week,Once a day,Few times a week,Few times a week,Once a day,Once a month,Once a month,Few times a week,Never,7-10 cups
18-24,Female,3,Eggetarian (Vegetarian who consumes egg and egg products),Sometimes,Sometimes,Never,Often,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Few times a week,Few times a week,Once a day,In every meal,Few times a week,Never,Never,Never,Never,Once a month,Once a month,Once a day,Once a month,4-6 cups
45-54,Female,3,Non-Vegetarian,Rarely,Sometimes,Rarely,Very frequently,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Once a month,In every meal,In every meal,In every meal,Once a month,Once a day,Once a month,Once a month,Few times a week,Once a month,Never,Never,Once a month,4-6 cups
Above 65,Male,3,Pescatarian (Vegetarian who consumes only seafood),Never,Rarely,Never,Never,Lunch,Freshly home-cooked produce,2,I do not have any allergies,Once a day,Several times a day,Once a day,In every meal,Once a day,Never,Several times a day,Once a day,Never,Few times a week,Never,Few times a week,Never,More than 15 cups
18-24,Male,2,Non-Vegetarian,Often,Often,Never,Sometimes,Lunch,Freshly home-cooked produce,1,I do not have any allergies,Several times a day,Once a day,Few times a week,Once a day,Several times a day,Once a day,Few times a week,Few times a week,Once a day,Once a month,Once a month,Few times a week,Never,4-6 cups
Under 18,Male,3,Eggetarian (Vegetarian who consumes egg and egg products),Never,Often,Never,Never,Lunch,Freshly home-cooked produce,2,I do not have any allergies,Few times a week,Once a day,In every meal,Several times a day,Several times a day,Never,Never,Never,Never,Once a month,Few times a week,Once a day,Never,4-6 cups
35-44,Female,3,Pescatarian (Vegetarian who consumes only seafood),Never,Sometimes,Never,Very frequently,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Few times a week,Few times a week,In every meal,In every meal,Few times a week,Once a month,Once a month,Once a day,Never,Never,Few times a week,Once a day,Never,7-10 cups
45-54,Female,2,Vegetarian (No egg or meat),Rarely,Sometimes,Never,Very frequently,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Few times a week,Few times a week,Several times a day,Several times a day,Few times a week,Never,Never,Once a day,Once a day,Never,Few times a week,Never,Never,7-10 cups
18-24,Male,4,Non-Vegetarian,Rarely,Never,Never,Rarely,Lunch,Freshly home-cooked produce,4,I do not have any allergies,Few times a week,Several times a day,Once a day,Once a day,Several times a day,Once a day,Once a month,Once a month,Once a month,Few times a week,Few times a week,Once a day,Once a month,11-14 cups
Above 65,Female,2,Vegetarian (No egg or meat),Sometimes,Never,Never,Very frequently,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Several times a day,Several times a day,Once a day,In every meal,Few times a week,Never,Never,Once a day,Never,Never,Once a month,Once a day,Never,11-14 cups
18-24,Female,3,Non-Vegetarian,Sometimes,Rarely,Never,Sometimes,Lunch,Freshly home-cooked produce,2,I do not have any allergies,Few times a week,Once a day,Few times a week,Once a day,Few times a week,Once a month,Once a month,Never,Once a day,Never,Never,Once a day,Once a month,7-10 cups
35-44,Female,2,Eggetarian (Vegetarian who consumes egg and egg products),Sometimes,Sometimes,Never,Sometimes,Lunch,Freshly home-cooked produce,1,I do not have any allergies,Once a day,In every meal,Several times a day,Several times a day,Few times a week,Never,Never,Several times a day,Never,Never,Few times a week,With every meal,Never,7-10 cups
Under 18,Female,3,Non-Vegetarian,Sometimes,Sometimes,Never,Rarely,Breakfast,Freshly home-cooked produce,1,I do not have any allergies,Few times a week,Few times a week,Once a day,Few times a week,Once a month,Few times a week,Few times a week,Never,Few times a week,Once a month,Once a day,Once a day,Never,4-6 cups
45-54,Male,3,Non-Vegetarian,Never,Sometimes,Rarely,Very frequently,Lunch,Freshly home-cooked produce,1,I do not have any allergies,Once a month,Once a day,Once a day,Once a day,Once a day,Once a day,Once a month,Once a day,Few times a week,Once a month,Never,Few times a week,Never,7-10 cups
18-24,Male,3,Non-Vegetarian,Rarely,Sometimes,Never,Sometimes,Lunch,Freshly home-cooked produce,1,I do not have any allergies,Once a month,Once a day,Once a day,Once a day,Few times a week,Once a month,Never,Few times a week,Few times a week,Once a month,Never,Few times a week,Never,7-10 cups"""


@st.cache_data(show_spinner="Veri okunuyor…")
def _load(file, demo: bool) -> pd.DataFrame:
    if demo or file is None:
        from io import StringIO
        return pd.read_csv(StringIO(DEMO_CSV))
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8-sig")
    return pd.read_excel(file)


@st.cache_data(show_spinner="Temizleniyor…")
def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_df(raw)


raw_df = _load(uploaded, use_demo)
df     = _clean(raw_df)

# Filtreler
age_opts  = sorted(df["age_raw"].dropna().unique().tolist())
gen_opts  = ["Tümü"] + sorted(df["gender"].dropna().unique().tolist())
diet_opts = ["Tümü"] + sorted(df["diet_short"].dropna().unique().tolist())

sel_ages   = _age_ph.multiselect("Yaş Grubu", age_opts, default=age_opts)
sel_gender = _gender_ph.selectbox("Cinsiyet", gen_opts)
sel_diet   = _diet_ph.selectbox("Diyet Tipi", diet_opts)

fdf = df.copy()
if sel_ages:
    fdf = fdf[fdf["age_raw"].isin(sel_ages)]
if sel_gender != "Tümü":
    fdf = fdf[fdf["gender"] == sel_gender]
if sel_diet != "Tümü":
    fdf = fdf[fdf["diet_short"] == sel_diet]

# ── Sekmeler ──────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🧹 Adım 1 · Veri Temizleme",
    "📊 Adım 2 · Analiz",
    "📱 Adım 3 · Post Üret",
    "📥 Adım 4 · Export",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADIM 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t1:
    st.markdown("## 🧹 Veri Temizleme Raporu")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ham Satır",    len(raw_df))
    c2.metric("Temiz Satır",  len(df))
    c3.metric("Sütun",        len(df.columns))
    c4.metric("Eksik Değer",  int(df.isnull().sum().sum()))

    st.markdown("---")
    st.markdown("### 🔄 Uygulanan Dönüşümler")
    st.dataframe(pd.DataFrame([
        ("Sütun yeniden adlandırma", "26 uzun sütun → kısa kodlar", "✅"),
        ("Yaş grubu → sayı",         "'18-24' → 21 (orta nokta)",   "✅"),
        ("Frekans metni → skor",     "'Several times a day' → 14",   "✅"),
        ("Su tüketimi → ml/gün",     "'7-10 cups' → 2125 ml",        "✅"),
        ("Diyet tipi kısaltma",      "Uzun isim → kısa etiket",       "✅"),
        ("Sağlık skoru (0-100)",     "Meyve+Sebze−Tatlı−Kızartma",  "✅"),
        ("Risk sınıflandırması",     "7 kategori kural tabanlı",      "✅"),
    ], columns=["Dönüşüm","Açıklama","Durum"]),
    use_container_width=True, hide_index=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Ham Veri")
        st.dataframe(raw_df.head(5), use_container_width=True, hide_index=True)
    with cb:
        st.markdown("#### Temiz Veri")
        show = [c for c in ["age_raw","age","gender","diet_short","health_score",
                             "top_risk","risk_count"] if c in df.columns]
        st.dataframe(df[show].head(5), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Frekans → Sayısal Skor Eşlemesi")
    st.dataframe(
        pd.DataFrame(sorted(FREQ_SCORE.items(), key=lambda x: x[1]),
                     columns=["Metin Değer","Skor"]),
        use_container_width=True, hide_index=True, height=280
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADIM 2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t2:
    st.markdown("## 📊 Veri Analizi")

    if fdf.empty:
        st.warning("Filtre sonucunda veri kalmadı.")
        st.stop()

    k = get_kpis(fdf)
    top_meta = RISK_META.get(k["top_risk"], {})

    cc1,cc2,cc3,cc4,cc5 = st.columns(5)
    cc1.metric("👥 Katılımcı",       str(k["n"]))
    cc2.metric("💚 Ort. Sağlık",     f"{k['avg_health']}/100")
    cc3.metric("⚠️ Çok Riskli",      f"%{k['pct_multi_risk']}")
    cc4.metric("❌ Diyetisyen Yok",  f"%{k['pct_no_consult']}")
    cc5.metric("🔴 Baskın Risk",     f"{top_meta.get('icon','')} {top_meta.get('label',k['top_risk'])}")

    st.markdown("---")
    _LAYOUT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                   font_color="#F1F5F9", margin=dict(t=10,b=10))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Yaş Grubu → Sağlık Skoru")
        d = fdf.groupby("age_raw")["health_score"].mean().reset_index()
        fig = px.bar(d, x="age_raw", y="health_score", color="health_score",
                     color_continuous_scale="RdYlGn", range_color=[40,80],
                     labels={"age_raw":"","health_score":"Ort. Sağlık"})
        fig.update_layout(**_LAYOUT, height=290, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("#### Risk Dağılımı")
        all_r = [r for rs in fdf["risks"] for r in rs]
        rc = Counter(all_r)
        fig2 = go.Figure(go.Pie(
            labels=[RISK_META.get(k,{}).get("label",k) for k in rc],
            values=list(rc.values()),
            marker_colors=[RISK_META.get(k,{}).get("color","#888") for k in rc],
            hole=0.42, textinfo="label+percent"))
        fig2.update_layout(**_LAYOUT, height=290, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("#### Diyet Tipi → Sağlık Skoru")
        d3 = fdf.groupby("diet_short")["health_score"].mean().sort_values().reset_index()
        fig3 = px.bar(d3, y="diet_short", x="health_score", orientation="h",
                      color="health_score", color_continuous_scale="Blues",
                      labels={"diet_short":"","health_score":"Sağlık Skoru"})
        fig3.update_layout(**_LAYOUT, height=290, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    with g4:
        st.markdown("#### Öğün Atlama → Sağlık Skoru")
        if "skip_meals" in fdf.columns:
            d4  = fdf.groupby("skip_meals")["health_score"].mean().reset_index()
            ord = ["Never","Rarely","Sometimes","Often","Very frequently"]
            d4["skip_meals"] = pd.Categorical(d4["skip_meals"], categories=ord, ordered=True)
            d4 = d4.sort_values("skip_meals")
            fig4 = px.line(d4, x="skip_meals", y="health_score", markers=True,
                           color_discrete_sequence=["#2563EB"],
                           labels={"skip_meals":"","health_score":"Sağlık"})
            fig4.update_layout(**_LAYOUT, height=290)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏥 Yaş Grubu × Dominant Risk")
    risk_df = top_risks_by_age(fdf)
    if not risk_df.empty:
        for _, row in risk_df[risk_df["rank"]==1].iterrows():
            st.markdown(f"""
            <div class="risk-row" style="border-left:4px solid {row['risk_color']}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <div style="font-size:.7rem;color:#64748B;text-transform:uppercase">{row['age_group']}</div>
                  <div style="font-weight:600;color:#F1F5F9">{row['risk_icon']} {row['risk_label']}</div>
                  <div style="font-size:.79rem;color:#94A3B8;margin-top:2px">{row['diet_tip']}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:1.35rem;font-weight:700;color:{row['risk_color']}">%{row['pct']}</div>
                  <div style="font-size:.7rem;color:#64748B">{row['count']}/{row['total']} kişi</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADIM 3 — POST ÜRETİMİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t3:
    st.markdown("## 📱 Sosyal Medya Post Üretimi")

    st.markdown('<div class="step-h">① Hedef Segment Seç</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    age_opts2 = sorted(fdf["age_raw"].dropna().unique().tolist())
    with p1:
        sel_age = st.selectbox("Yaş Grubu", age_opts2, key="pa")
    risk_df2 = top_risks_by_age(fdf)
    seg_risks = (risk_df2[risk_df2["age_group"]==sel_age]["risk_key"].tolist()
                 if not risk_df2.empty else [])
    all_rk = list(RISK_META.keys())
    risk_opts = seg_risks + [r for r in all_rk if r not in seg_risks]
    with p2:
        sel_risk = st.selectbox("Risk / Konu", risk_opts,
                                format_func=lambda x: f"{RISK_META.get(x,{}).get('icon','')} {RISK_META.get(x,{}).get('label',x)}")
    with p3:
        sel_gp = st.selectbox("Hedef Cinsiyet", ["Karma","Kadın","Erkek"])

    seg_pts = fdf[fdf["age_raw"]==sel_age]
    if sel_gp != "Karma" and "gender" in seg_pts.columns:
        seg_pts = seg_pts[seg_pts["gender"]==sel_gp]
    meta     = RISK_META.get(sel_risk, {})
    r_color  = meta.get("color","#2563EB")
    r_cnt    = int((seg_pts["risks"].apply(lambda rs: sel_risk in rs)).sum())
    avg_h    = round(seg_pts["health_score"].mean(), 1) if len(seg_pts) else 0

    st.markdown(f"""
    <div style="background:#1E293B;border:1px solid #334155;
                border-left:4px solid {r_color};border-radius:8px;
                padding:.9rem 1.1rem;margin:.5rem 0 1rem">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem">
        <div><div style="font-size:.68rem;color:#64748B;text-transform:uppercase">Segment</div>
             <div style="font-weight:600;color:#F1F5F9">{sel_age} · {sel_gp}</div></div>
        <div><div style="font-size:.68rem;color:#64748B;text-transform:uppercase">Kişi</div>
             <div style="font-weight:600;color:#F1F5F9">{len(seg_pts)}</div></div>
        <div><div style="font-size:.68rem;color:#64748B;text-transform:uppercase">Risk Oranı</div>
             <div style="font-weight:600;color:{r_color}">{meta.get('icon','')} {r_cnt}/{len(seg_pts)}</div></div>
        <div><div style="font-size:.68rem;color:#64748B;text-transform:uppercase">Sağlık</div>
             <div style="font-weight:600;color:#F1F5F9">{avg_h}/100</div></div>
      </div>
      <div style="margin-top:7px;font-size:.81rem;color:#94A3B8">
        <b style="color:#CBD5E1">Diyet İpucu:</b> {meta.get('diet_tip','')}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="step-h">② Platform & Format</div>', unsafe_allow_html=True)
    st.info(f"**Platform:** {platform}  ·  **Format:** {POST_TYPES.get(post_type,'')}  ·  "
            f"**Karakter limiti:** {CHAR_LIMIT.get(platform,500)}  — Sol panelden değiştirebilirsiniz.")

    st.markdown('<div class="step-h">③ İçerik Üret</div>', unsafe_allow_html=True)
    has_ai = bool(os.environ.get("ANTHROPIC_API_KEY",""))
    b1, b2 = st.columns(2)
    with b1:
        gen_tmpl = st.button("📝 Şablon ile Üret", use_container_width=True)
    with b2:
        gen_ai   = st.button("🤖 AI ile Üret (Claude)", use_container_width=True,
                             disabled=not has_ai,
                             help="Sol panelden API key ekleyin" if not has_ai else "")

    if gen_tmpl:
        st.session_state["post"]   = decide_post(sel_age, sel_risk, platform, post_type, len(seg_pts), avg_h)
        st.session_state["is_ai"]  = False

    if gen_ai and has_ai:
        with st.spinner("Claude yazıyor…"):
            res = generate_ai_post({"age_group":sel_age,"risk_key":sel_risk,
                                    "patient_count":len(seg_pts),"avg_health":avg_h},
                                   platform, post_type)
            st.session_state["post"]  = res or decide_post(sel_age, sel_risk, platform, post_type, len(seg_pts), avg_h)
            st.session_state["is_ai"] = bool(res)

    if "post" in st.session_state:
        c = st.session_state["post"]
        is_ai = st.session_state.get("is_ai", False)
        ht = " ".join(f'<span class="hashtag">{h}</span>' for h in c.get("hashtags",[]))
        bc = "#1D4ED8" if is_ai else "#374151"
        bt = "🤖 AI" if is_ai else "📝 Şablon"

        st.markdown(f"""
        <div class="post-card" style="border-top-color:{r_color}">
          <div style="display:flex;justify-content:space-between;margin-bottom:7px">
            <div class="post-title">{c.get('title','')}</div>
            <span style="background:{bc};color:white;font-size:.69rem;
                         padding:2px 9px;border-radius:20px">{bt} · {platform}</span>
          </div>
          <div class="post-caption">{c.get('caption','').replace(chr(10),'<br>')}</div>
          <div style="margin:5px 0">{ht}</div>
          <div style="margin-top:7px">
            <span class="pill">⏰ {c.get('best_time','')}</span>
            <span class="pill">🖼️ {c.get('visual_tip','')[:55]}…</span>
          </div>
          <div style="border-top:1px solid #334155;margin-top:9px;padding-top:7px;
                      font-size:.78rem;color:#64748B">💡 {c.get('why','')}</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("✏️ Metni Düzenle / Kopyala"):
            txt = f"{c.get('title','')}\n\n{c.get('caption','')}\n\n" + " ".join(c.get("hashtags",[]))
            edited = st.text_area("Gönderi Metni", txt, height=240, key="edit_post")
            limit = CHAR_LIMIT.get(platform, 500)
            st.caption(f"Karakter: {len(edited)} / {limit}")
            if len(edited) > limit:
                st.warning(f"⚠️ {platform} sınırını ({limit}) aşıyor!")

        with st.expander("🎨 Görsel Rehber"):
            st.markdown(f"**Görsel:** {c.get('visual_tip','')}\n\n**CTA:** _{c.get('cta','')}_")

    # Haftalık plan
    st.markdown("---")
    st.markdown('<div class="step-h">④ Haftalık İçerik Takvimi</div>', unsafe_allow_html=True)

    if st.button("📅 Haftalık Plan Oluştur", use_container_width=True):
        rows = []
        days  = ["Pazartesi","Çarşamba","Cuma","Pazar"]
        types = ["carousel","story","reel","infographic"]
        segs  = (risk_df2[risk_df2["age_group"]==sel_age].head(4)
                 if not risk_df2.empty else pd.DataFrame())
        for i, (_, row) in enumerate(segs.iterrows()):
            con = decide_post(row["age_group"], row["risk_key"], platform,
                              types[i%4], int(row["total"]),
                              round(fdf["health_score"].mean(),1))
            rows.append({
                "Gün": days[i%4], "Yaş Grubu": row["age_group"],
                "Risk": row["risk_label"], "Format": POST_TYPES[types[i%4]],
                "Başlık": con["title"],
                "Caption": con["caption"][:150]+"…",
                "Hashtags": " ".join(con["hashtags"]),
                "En İyi Saat": con["best_time"],
            })
        if rows:
            plan = pd.DataFrame(rows)
            st.dataframe(plan, use_container_width=True, hide_index=True)
            st.session_state["plan_df"] = plan
        else:
            st.info("Segment için yeterli risk verisi yok.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADIM 4 — EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t4:
    st.markdown("## 📥 Export")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("#### Temiz Veri")
        st.download_button("⬇️ Excel", data=df_to_excel(fdf),
                           file_name="temiz_veri.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        st.download_button("⬇️ CSV",   data=fdf.to_csv(index=False, encoding="utf-8-sig").encode(),
                           file_name="temiz_veri.csv", mime="text/csv",
                           use_container_width=True)

    with e2:
        st.markdown("#### Risk Analizi")
        rdf = top_risks_by_age(fdf)
        if not rdf.empty:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                rdf.to_excel(w, index=False, sheet_name="Riskler")
            st.download_button("⬇️ Excel", data=buf.getvalue(),
                               file_name="risk_analizi.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    with e3:
        st.markdown("#### Sosyal Medya Takvimi")
        if "plan_df" in st.session_state:
            buf2 = BytesIO()
            with pd.ExcelWriter(buf2, engine="xlsxwriter") as w:
                st.session_state["plan_df"].to_excel(w, index=False, sheet_name="Takvim")
            st.download_button("⬇️ Excel", data=buf2.getvalue(),
                               file_name="icerik_takvimi.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        else:
            st.info("Önce Adım 3'te haftalık plan oluşturun.")

    st.markdown("---")
    st.markdown("#### 👁️ Tam Veri Tablosu")
    scols = [c for c in fdf.columns
             if not c.endswith("_s") and c != "age_raw"
             and c not in [f+"_s" for f in FREQ_COLS]]
    st.dataframe(fdf[scols], use_container_width=True, hide_index=True, height=400)
