import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="Pro AI Cricket Predictor", page_icon="🏆", layout="wide")
st.title("🏆 Pro AI Cricket Win Predictor (Pitch Factor + Live API)")

# ১. আন্তর্জাতিক স্টেডিয়াম ও পিচ প্রোফাইল (গড় রান ও চেজিং ডিফিকাল্টি ফ্যাক্টর)
VENUES = {
    "Mirpur (Sher-e-Bangla) - Slow/Spin": {"avg_score": 145, "chase_bias": -0.18},
    "Chattogram (ZAC Stadium) - Batting Friendly": {"avg_score": 175, "chase_bias": 0.12},
    "Mumbai (Wankhede) - High Scoring": {"avg_score": 192, "chase_bias": 0.22},
    "Chennai (Chepauk) - Spin Friendly": {"avg_score": 158, "chase_bias": -0.12},
    "Kolkata (Eden Gardens) - Balanced/Fast Outfield": {"avg_score": 178, "chase_bias": 0.10},
    "Melbourne (MCG) - Big Boundaries": {"avg_score": 165, "chase_bias": -0.05},
    "Dubai International Stadium - Dew Factor": {"avg_score": 168, "chase_bias": 0.15},
    "Neutral / Other Venue": {"avg_score": 170, "chase_bias": 0.0}
}

# ২. উন্নত AI মডেল ট্রেইনিং (Random Forest + Venue Encoding)
@st.cache_resource
def train_advanced_model():
    np.random.seed(42)
    n_samples = 6000
    
    venue_list = list(VENUES.keys())
    selected_venues = np.random.choice(venue_list, n_samples)
    
    runs_left = np.random.randint(1, 180, n_samples)
    balls_left = np.random.randint(6, 120, n_samples)
    wickets_left = np.random.randint(1, 11, n_samples)
    crr = np.random.uniform(4.5, 13.0, n_samples)
    rrr = (runs_left * 6) / balls_left
    
    # পিচের আচরণ ও ডেথ ওভারের চাপ হিসাব করে বাস্তবসম্মত ফলাফল তৈরি
    results = []
    for i in range(n_samples):
        v_bias = VENUES[selected_venues[i]]["chase_bias"]
        # শেষ ৫ ওভারে RRR ১০-এর বেশি হলে জেতা কঠিন
        death_over_penalty = -1.5 if (balls_left[i] <= 30 and rrr[i] > 10.5) else 0.0
        
        win_metric = (
            (wickets_left[i] * 0.48) 
            - (rrr[i] * 0.38) 
            + (crr[i] * 0.12) 
            - (runs_left[i] * 0.015) 
            + (v_bias * 4.5) 
            + death_over_penalty
        )
        prob = 1 / (1 + np.exp(-win_metric))
        results.append(1 if prob > 0.5 else 0)
        
    df = pd.DataFrame({
        'venue': selected_venues,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'crr': crr,
        'rrr': rrr,
        'result': results
    })
    
    X = df[['venue', 'runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr']]
    y = df['result']
    
    # ক্যাটাগরিকাল ডেটা (Venue) ও নিউমেরিক্যাল ডেটার পাইপলাইন
    preprocessor = ColumnTransformer(
        transformers=[('venue_enc', OneHotEncoder(handle_unknown='ignore'), ['venue'])],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

model = train_advanced_model()

# ৩. সাইডবার কন্ট্রোল ও ভেন্যু নির্বাচন
st.sidebar.header("⚙️ ম্যাচ ও পিচ সেটিংস")
mode = st.sidebar.radio("ডেটা সোর্স:", ["ম্যানুয়াল স্লাইডার (Manual)", "লাইভ ম্যাচ API (CricAPI)"])
selected_venue = st.sidebar.selectbox("🏟️ স্টেডিয়াম / পিচের ধরন:", list(VENUES.keys()))

venue_avg = VENUES[selected_venue]["avg_score"]
st.sidebar.caption(f"এই পিচের গড় ১ম ইনিংস স্কোর: **{venue_avg}**")

batting_team, bowling_team = "Bangladesh", "India"
target, current_score, overs_completed, wickets_lost = venue_avg + 10, 115, 13.2, 3

if mode == "লাইভ ম্যাচ API (CricAPI)":
    api_key = st.sidebar.text_input("আপনার CricAPI Key দিন:", type="password")
    if api_key:
        try:
            url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
            res = requests.get(url).json()
            if res.get("status") == "success":
                t20s = [m for m in res.get("data", []) if m.get("matchType") == "t20" and m.get("matchStarted")]
                if t20s:
                    m_name = st.sidebar.selectbox("চলমান T20 ম্যাচ:", [m["name"] for m in t20s])
                    m_data = next(m for m in t20s if m["name"] == m_name)
                    scores = m_data.get("score", [])
                    teams = m_data.get("teams", ["Team A", "Team B"])
                    st.sidebar.success(f"স্ট্যাটাস: {m_data.get('status')}")
                    
                    if len(scores) >= 2:
                        target = scores[0].get("r", 160) + 1
                        current_score = scores[1].get("r", 0)
                        wickets_lost = min(9, scores[1].get("w", 0))
                        overs_completed = float(scores[1].get("o", 1.0))
                        batting_team, bowling_team = teams[1], teams[0]
                    elif len(scores) == 1:
                        st.sidebar.info("১ম ইনিংস চলছে! ২য় ইনিংসের চেজ সিমুলেট করতে টার্গেট ঠিক করুন:")
                        current_score = scores[0].get("r", 0)
                        wickets_lost = min(9, scores[0].get("w", 0))
                        overs_completed = float(scores[0].get("o", 1.0))
                        target = st.sidebar.slider("সম্ভাব্য টার্গেট:", current_score + 10, 260, venue_avg)
                else:
                    st.sidebar.warning("কোনো লাইভ T20 ম্যাচ চলছে না। ম্যানুয়াল মোড ব্যবহার করুন।")
            else:
                st.sidebar.error("API Key কাজ করছে না বা লিমিট শেষ!")
        except Exception as e:
            st.sidebar.error(f"ত্রুটি: {e}")
else:
    batting_team = st.sidebar.text_input("ব্যাটিং দল", "Bangladesh")
    bowling_team = st.sidebar.text_input("বোলিং দল", "India")
    target = st.sidebar.slider("টার্গেট স্কোর (Target)", 100, 260, 180)
    current_score = st.sidebar.slider("বর্তমান রান (Current Score)", 0, target, 120)
    overs_completed = st.sidebar.slider("সম্পন্ন ওভার (Overs)", 0.1, 19.5, 14.0, 0.1)
    wickets_lost = st.sidebar.slider("পতন হওয়া উইকেট (Wickets)", 0, 9, 3)

# ৪. ক্রিকেট ওভার ও বল ক্যালকুলেশন
whole_overs = int(overs_completed)
balls_in_over = min(5, int(round((overs_completed - whole_overs) * 10)))
balls_played = max(1, (whole_overs * 6) + balls_in_over)

runs_left = max(0, target - current_score)
balls_left = max(1, 120 - balls_played)
wickets_left = 10 - wickets_lost

crr = round((current_score * 6) / balls_played, 2)
rrr = round((runs_left * 6) / balls_left, 2)

# ৫. AI প্রেডিকশন (পিচ ফ্যাক্টরসহ)
if runs_left == 0:
    win_prob, loss_prob = 100.0, 0.0
else:
    input_df = pd.DataFrame([[selected_venue, runs_left, balls_left, wickets_left, crr, rrr]],
                            columns=['venue', 'runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr'])
    probs = model.predict_proba(input_df)[0]
    loss_prob = round(probs[0] * 100, 1)
    win_prob = round(probs[1] * 100, 1)

# ৬. ড্যাশবোর্ড ভিউ
col1, col2, col3, col4 = st.columns(4)
col1.metric("স্কোরবোর্ড", f"{current_score}/{wickets_lost} ({whole_overs}.{balls_in_over})", f"টার্গেট: {target}")
col2.metric("সমীকরণ", f"{runs_left} রান দরকার", f"{balls_left} বল বাকি")
col3.metric("Current Run Rate", f"{crr}", f"পিচ গড়: {venue_avg}")
col4.metric("Required Run Rate", f"{rrr}", delta=round(crr - rrr, 2))

st.divider()

# ৭. গেজ চার্ট ও প্রোবাবিলিটি বার
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 ম্যাচ জয়ের সম্ভাব্যতা (AI Win Probability)")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['Win %'], x=[win_prob], name=batting_team, orientation='h',
        marker=dict(color='#00C853'), text=f"{batting_team}: {win_prob}%",
        textposition='inside', insidetextanchor='middle', textfont=dict(size=16, color='white')
    ))
    fig_bar.add_trace(go.Bar(
        y=['Win %'], x=[loss_prob], name=bowling_team, orientation='h',
        marker=dict(color='#D50000'), text=f"{bowling_team}: {loss_prob}%",
        textposition='inside', insidetextanchor='middle', textfont=dict(size=16, color='white')
    ))
    fig_bar.update_layout(
        barmode='stack', height=140, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], showticklabels=False), yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # AI Match Insight Box
    st.info(
        f"🧠 **AI ম্যাচ বিশ্লেষণ:** **{selected_venue.split('-')[0]}**-এর পিচে জয়ের জন্য প্রতি ওভারে **{rrr}** রান প্রয়োজন। "
        f"হাতে **{wickets_left}** উইকেট থাকায় এই মুহূর্তে **{batting_team if win_prob >= 50 else bowling_team}** এগিয়ে আছে।"
    )

with c2:
    # প্রেসার ইনডেক্স মিটার (Pressure Gauge)
    pressure_score = min(100, max(0, int((rrr * 5) + ((10 - wickets_left) * 6))))
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pressure_score,
        title={'text': "ব্যাটিং চাপ (Pressure Index)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF6D00"},
            'steps': [
                {'range': [0, 40], 'color': "#E8F5E9"},
                {'range': [40, 70], 'color': "#FFF3E0"},
                {'range': [70, 100], 'color': "#FFEBEE"}
            ]
        }
    ))
    fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
