import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

st.set_page_config(page_title="AI Cricket Win Predictor", page_icon="🏏", layout="wide")
st.title("🏏 AI T20 Cricket Win Probability (Live API + Manual)")

# ১. মডেল ট্রেইনিং
@st.cache_resource
def train_prediction_model():
    np.random.seed(42)
    n_samples = 2500
    runs_left = np.random.randint(1, 160, n_samples)
    balls_left = np.random.randint(6, 120, n_samples)
    wickets_left = np.random.randint(1, 11, n_samples)
    crr = np.random.uniform(5.0, 12.0, n_samples)
    rrr = (runs_left * 6) / balls_left
    
    score_metric = (wickets_left * 4.5) - (rrr * 3.2) + (crr * 1.2) - (runs_left * 0.12)
    prob = 1 / (1 + np.exp(-score_metric / 5))
    result = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'runs_left': runs_left, 'balls_left': balls_left,
        'wickets_left': wickets_left, 'crr': crr, 'rrr': rrr, 'result': result
    })
    
    model = LogisticRegression()
    model.fit(df[['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr']], df['result'])
    return model

model = train_prediction_model()

# ২. সাইডবারে মোড নির্বাচন (Live API নাকি Manual Slider)
st.sidebar.header("⚙️ ইনপুট মোড নির্বাচন করুন")
mode = st.sidebar.radio("ডেটা সোর্স:", ["ম্যানুয়াল স্লাইডার (Manual)", "লাইভ ম্যাচ API (CricAPI)"])

# ডিফল্ট ভ্যালু
batting_team, bowling_team = "Bangladesh", "India"
target, current_score, overs_completed, wickets_lost = 185, 128, 14.0, 3

if mode == "লাইভ ম্যাচ API (CricAPI)":
    api_key = st.sidebar.text_input("আপনার CricAPI Key দিন:", type="password")
    
    if api_key:
        try:
            url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
            response = requests.get(url).json()
            
            if response.get("status") == "success":
                matches = response.get("data", [])
                t20_matches = [m for m in matches if m.get("matchType") == "t20" and m.get("matchStarted")]
                
                if t20_matches:
                    match_names = [m["name"] for m in t20_matches]
                    selected_match_name = st.sidebar.selectbox("চলমান T20 ম্যাচ বেছে নিন:", match_names)
                    
                    # নির্বাচিত ম্যাচের ডেটা বের করা
                    match_data = next(m for m in t20_matches if m["name"] == selected_match_name)
                    scores = match_data.get("score", [])
                    teams = match_data.get("teams", ["Team A", "Team B"])
                    
                    st.sidebar.info(f"স্ট্যাটাস: {match_data.get('status')}")
                    
                    if len(scores) >= 2:
                        # ২য় ইনিংস চলছে (টার্গেট চেজ)
                        inn1, inn2 = scores[0], scores[1]
                        target = inn1.get("r", 150) + 1
                        current_score = inn2.get("r", 0)
                        wickets_lost = min(9, inn2.get("w", 0))
                        overs_completed = float(inn2.get("o", 1.0))
                        batting_team = teams[1]
                        bowling_team = teams[0]
                    elif len(scores) == 1:
                        st.sidebar.warning("এখন ১ম ইনিংস চলছে! ২য় ইনিংসের প্রেডিকশন দেখতে সম্ভাব্য টার্গেট স্লাইডারে সেট করুন।")
                        inn1 = scores[0]
                        current_score = inn1.get("r", 0)
                        wickets_lost = min(9, inn1.get("w", 0))
                        overs_completed = float(inn1.get("o", 1.0))
                        target = st.sidebar.slider("সম্ভাব্য টার্গেট ধরুন:", current_score + 1, 260, 180)
                else:
                    st.sidebar.warning("এই মুহূর্তে কোনো লাইভ T20 ম্যাচ পাওয়া যায়নি। ম্যানুয়াল স্লাইডার ব্যবহার করুন।")
            else:
                st.sidebar.error("API Key সঠিক নয় অথবা আজকের লিমিট শেষ!")
        except Exception as e:
            st.sidebar.error(f"ডেটা লোড করতে সমস্যা হয়েছে: {e}")
    else:
        st.sidebar.info("লাইভ স্কোর আনতে ওপরে আপনার API Key পেস্ট করুন।")

else:
    # ম্যানুয়াল স্লাইডার মোড
    batting_team = st.sidebar.text_input("ব্যাটিং দল", "Bangladesh")
    bowling_team = st.sidebar.text_input("বোলিং দল", "India")
    target = st.sidebar.slider("টার্গেট স্কোর (Target)", 100, 260, 185)
    current_score = st.sidebar.slider("বর্তমান রান (Current Score)", 0, target, 128)
    overs_completed = st.sidebar.slider("সম্পন্ন ওভার (Overs)", 0.1, 19.5, 14.0, 0.1)
    wickets_lost = st.sidebar.slider("পতন হওয়া উইকেট (Wickets)", 0, 9, 3)

# ৩. গাণিতিক হিসাব (ওভারের বল রূপান্তর: যেমন 14.2 মানে ১৪ ওভার ২ বল)
whole_overs = int(overs_completed)
balls_in_over = int(round((overs_completed - whole_overs) * 10))
if balls_in_over > 5:
    balls_in_over = 5
balls_played = max(1, (whole_overs * 6) + balls_in_over)

runs_left = max(0, target - current_score)
balls_left = max(1, 120 - balls_played)
wickets_left = 10 - wickets_lost

crr = round((current_score * 6) / balls_played, 2)
rrr = round((runs_left * 6) / balls_left, 2)

# ৪. প্রেডিকশন বের করা
if runs_left == 0:
    win_prob, loss_prob = 100.0, 0.0
else:
    input_df = pd.DataFrame([[runs_left, balls_left, wickets_left, crr, rrr]],
                            columns=['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr'])
    probs = model.predict_proba(input_df)[0]
    loss_prob = round(probs[0] * 100, 1)
    win_prob = round(probs[1] * 100, 1)

# ৫. ড্যাশবোর্ড মেট্রিক্স ও গ্রাফ
col1, col2, col3, col4 = st.columns(4)
col1.metric("স্কোরবোর্ড", f"{current_score}/{wickets_lost} ({overs_completed})", f"টার্গেট: {target}")
col2.metric("জয়ের সমীকরণ", f"{runs_left} রান", f"{balls_left} বল বাকি")
col3.metric("Current Run Rate (CRR)", f"{crr}")
col4.metric("Required Run Rate (RRR)", f"{rrr}", delta=round(crr - rrr, 2))

st.divider()
st.subheader("📊 জয়ের সম্ভাব্যতা (Real-Time Win Probability)")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=['Win %'], x=[win_prob], name=batting_team, orientation='h',
    marker=dict(color='#00C853'), text=f"{batting_team}: {win_prob}%",
    textposition='inside', insidetextanchor='middle', textfont=dict(size=16, color='white')
))
fig_bar.add_trace(go.Bar(
    y=['Win %'], x=[loss_prob], name=bowling_team, orientation='h',
    marker=dict(color='#FF3D00'), text=f"{bowling_team}: {loss_prob}%",
    textposition='inside', insidetextanchor='middle', textfont=dict(size=16, color='white')
))
fig_bar.update_layout(
    barmode='stack', height=130, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[0, 100], showticklabels=False), yaxis=dict(showticklabels=False)
)
st.plotly_chart(fig_bar, use_container_width=True)
