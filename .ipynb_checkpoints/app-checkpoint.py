import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import types
from pathlib import Path
import base64
#define weighted_logloss

def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    return grad, hess

# add in the original training module name:
mod = types.ModuleType("model_train")
mod.weighted_logloss = weighted_logloss
sys.modules["model_train"] = mod

# Streamlit page setup 
st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

icon_base64 = img_to_base64("figures/title_icon2.png")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    body {{ font-family: 'Poppins', sans-serif; background-color: #f4f6f8; }}

    nav.navbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 2rem;
      background-color: #1E3A8A;  /* solid blue */
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      margin-bottom: 1.5rem;
    }}

    .navbar-logo {{
      display: inline-flex;
      align-items: center;
      color: #0000FF !important;   /* force white text */
      font-size: 3.5rem;            /* larger text */
      font-weight: 800;
      letter-spacing: 0.12em;
      text-shadow: none;
      transition: filter 0.3s ease;
    }}
    .navbar-logo img {{
      width: 64px;   /* larger icon */
      height: 64px;
      object-fit: contain;
      margin-right: 1rem;
      border-radius: 8px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }}

    .navbar-logo:hover {{
      filter: brightness(1.1) drop-shadow(0 0 6px #FFFFFF);
      cursor: pointer;
    }}

    hr.divider {{
      border: none;
      height: 4px;
      background: #FFFFFF;
      border-radius: 2px;
      margin: 1.5rem 0;
    }}
    </style>

    <nav class="navbar">
      <div class="navbar-logo">
        <img src="data:image/png;base64,{icon_base64}" alt="EmpowerHER Logo">
        <span>EMPOWERHER</span>
      </div>
    </nav>

    <hr class="divider"/>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["About", "Breast Cancer Risk Prediction", "Mind & Move"])

# load model
BASE_DIR = Path(__file__).resolve().parent
model     = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# Tab 1: About 
with tab1:
    st.markdown("### 📊 About this Breast Cancer Risk Model")
    st.info(***Research & Education use only:** This tool is not a medical diagnosis.")
    st.markdown("""
**XGBoost** is a state-of-the-art tree-based model for tabular data, capturing complex feature interactions. The model was trained on the Brest Cancer Surveillance Consortium **BCSC** cohort https://www.bcsc-research.org/index.php/datasets/rf/documentation with a custom weighted log-loss that was penalizing missed cancers much more heavily than false alarms to prioritize cancer detection. Predictions may be less reliable for populations under-represented e.g. certain ethnic groups, uninsured women, different health-care systems.
 ****This is a research model, not a medical diagnosis. Please consult your doctor****.

***How we know this breast cancer risk model is any good***
When you hear things like “89% accuracy” or “52% recall,” it can feel like jargon. Here’s what those numbers really mean and why we chose them, what they tell us about the model, and why they matter for you.
***Overall Accuracy***
Think of accuracy like a simple “right vs. wrong” score. If the model makes 100 predictions about who might develop breast cancer and who won’t—and 89 of those guesses match reality—that’s 89% accuracy.
***Recall***: Catching the real cases
If 100 women truly had early‐stage cancer and our model flagged 52 of them, recall is 52%.

""")
    st.image("figures/empowerher_risk_pipeline_clean.png", width=900)
    st.markdown("Users can select demographic and clinical data to see the model risk prediction.")
    st.image("figures/feature_importance_xgb.png", width=900)
    st.markdown("This plot shows the top predictors the model relies on.")
    st.image("figures/P-R chart2.png", width=900)
    st.markdown("Precision–Recall curve for this XGBoost classifier model.")
# Tab 2: Risk Insights 
with tab2:
    st.markdown("<h2>Breast Cancer Risk Prediction</h2>", unsafe_allow_html=True)
    st.info("**Research & Education Use Only:** Not for medical diagnosis.")
    with st.expander("Enter information", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])
        age_groups  = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",6:"50–54",
                       7:"55–59",8:"60–64",9:"65–69",10:"70–74",11:"75–79",12:"80–84",13:">85"}
        race_eth    = {1:"White",2:"Black",3:"Asian/Pacific",4:"Native",5:"Hispanic",6:"Other"}
        menarche    = {0:">14",1:"12–13",2:"<12"}
        birth_age   = {0:"<20",1:"20–24",2:"25–29",3:">30",4:"Nulliparous"}
        fam_hist    = {0:"No",1:"Yes"}
        biopsy      = {0:"No",1:"Yes"}
        density     = {1:"Almost fat",2:"Scattered",3:"Hetero-dense",4:"Extremely dense"}
        hormone_use = {0:"No",1:"Yes"}
        menopause   = {1:"Pre/peri",2:"Post",3:"Surgical"}
        bmi_group   = {1:"10–24.9",2:"25–29.9",3:"30–34.9",4:"35+"}
        inputs = {
           "age_group":         sel("Age group", age_groups),
            "race_eth":          sel("Race/Ethnicity", race_eth),
            "age_menarche":      sel("Age at 1st period", menarche),
            "age_first_birth":   sel("Age at first birth", birth_age),
            "family_history":    sel("Family history", fam_hist),
            "personal_biopsy":   sel("Personal biopsy history", biopsy),
            "density":           sel("BI-RADS density", density),
            "hormone_use":       sel("Hormone use", hormone_use),
            "menopausal_status": sel("Menopausal status", menopause),
            "bmi_group":         sel("BMI group", bmi_group),
        }

    raw_df = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new   = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)
    prob = model.predict_proba(df_new)[0,1]
    st.markdown(f"<h3>Estimated Probability of Breast Cancer: {prob:.1%}</h3>", unsafe_allow_html=True)
        
#Tab 3: Mind & Move ─────────────────────────────────────────────────────
with tab3:
    st.header("Glow and Grow")
    st.write(
        "Every healthy choice you make today is a step toward "
        "a brighter, happier you. You’ve got this!"
    )

    # Daily Rituals
    st.subheader("Daily Rituals")
    for tip in [
        "🧘 Practice 10 min mindfulness",
        "🥗 Eat ≥5 servings fruits/veggies",
        "🚶‍♀️ Take a 30 min walk",
        "💧 Drink 8 glasses of water",
        "😴 Get 7–8 h sleep",
    ]:
        st.markdown(f"- {tip}")

    # Live‐updating Tracker
    st.subheader("Tracker")
    c1, c2, c3 = st.columns(3)
    with c1:
        med = st.number_input("Meditation (min)", 0, 60, 0, key="med")
        st.progress(med / 10, text=f"{med} / 10 min")
        st.metric("Meditation", f"{med} min", f"{10-med} to goal")
    with c2:
        ex = st.number_input("Exercise (min)", 0, 180, 0, key="ex")
        st.progress(ex / 30, text=f"{ex} / 30 min")
        st.metric("Exercise", f"{ex} min", f"{30-ex} to goal")
    with c3:
        water = st.number_input("Water (glasses)", 0, 20, 0, key="water")
        st.progress(water / 8, text=f"{water} / 8 glasses")
        st.metric("Hydration", f"{water} glasses", f"{8-water} to goal")

    # Compute overall + mood
    overall = np.mean([med/10, ex/30, water/8])
    if overall == 0:
        mood = "😔"
    elif overall < 0.4:
        mood = "😐"
    elif overall < 0.6:
        mood = "🙂"
    elif overall < 0.8:
        mood = "😃"
    else:
        mood = "🤩"

    # Show as a metric: big number + emoji
    st.metric(
        label="Overall Wellness Progress",
        value=f"{overall*100:.0f}%",
        delta=mood
    )

    # Save button
    if st.button("Save Entry"):
        entry = {
            "Meditation (min)": med,
            "Exercise (min)":   ex,
            "Water (glasses)":  water,
            "Timestamp":        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Mood":             mood,
        }
        st.success("✅ Your daily wellness entry has been recorded!")
        st.json(entry)
    st.title("Additional Resources")
    videos = {
    "Mindfulness Meditation for Cancer Support": "https://www.youtube.com/watch?v=1ZYbU82GVz4&t=31s",
    "Gentle Move for All":                     "https://www.youtube.com/watch?v=Ev6yE55kYGw&t=169s",
    "Healthy Eating During Cancer Treatment":   "https://www.youtube.com/watch?v=VaVC3PAWqLk&t=1353s"
}

    st.markdown("## 🎥 YouTube Videos")
    cols = st.columns(len(videos))
    for col, (title, url) in zip(cols, videos.items()):
        col.video(url)
        col.caption(f"**{title}**")


    st.markdown("## 🤝 Local Support Groups")
    support = [
      {"Name": "American Cancer Society Middle Tennessee",            "Phone": "(800) 227-2345 (toll free)", "Website": "https://www.cancer.org/about-us/local/tennessee.html"},
     {"Name": "Tennessee Breast Cancer Coalition",            "Phone": "(615) 377-8777", "Website": "https://www.tbcc.org/"},
    {"Name": "Susan G. Komen Nashville",            "Phone": "(615) 673-6633", "Website": "https://komen.org/nashville"},
    {"Name": "Vanderbilt Breast Cancer Support Group","Phone": "(615) 322-3900", "Website": "https://www.vanderbilthealth.com/service-line/breast-center"},
    {"Name": "Alive Hospice Cancer Support",        "Phone": "(615) 327-1085", "Website": "https://alivehospice.org"},
    {"Name": "YMCA of Middle Tennessee – LIVESTRONG® at the Y", "Website": "https://www.ymcamidtn.org/programs/health-and-fitness/support-groups/after-breast-cancer"}
]
    support_df = pd.DataFrame(support)
    support_df["Website"] = support_df["Website"].apply(lambda url: f"[Visit]({url})")
    st.table(support_df)