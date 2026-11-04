import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="All-in-One AI Cricket Predictor", page_icon="🏆", layout="wide")
st.title("🏆 All-in-One AI Cricket Analytics & Fantasy Hub")

# ১. স্টেডিয়াম ও পিচ প্রোফাইল ডেটাবেস
VENUES = {
    "Mirpur (Sher-e-Bangla) - Slow/Spin": {"avg_score": 145, "chase_bias": -0.18, "spin_aid": 85, "pace_aid": 40},
    "Chattogram (ZAC Stadium) - Batting Friendly": {"avg_score": 175, "chase_bias": 0.12, "spin_aid": 50, "pace_aid": 60},
    "Mumbai (Wankhede) - High Scoring": {"avg_score": 192, "chase_bias": 0.22, "spin_aid": 35, "pace_aid": 75},
    "Chennai (Chepauk) - Spin Friendly": {"avg_score": 158, "chase_bias": -0.12, "spin_aid": 88, "pace_aid": 45},
    "Kolkata (Eden Gardens) - Fast Outfield": {"avg_score": 178, "chase_bias": 0.10, "spin_aid": 55, "pace_aid": 70},
    "Melbourne (MCG) - Big Boundaries/Pace": {"avg_score": 165, "chase_bias": -0.05, "spin_aid": 40, "pace_aid": 85},
    "Dubai International Stadium - Dew Factor": {"avg_score": 168, "chase_bias": 0.18, "spin_aid": 60, "pace_aid": 65}
}

# ২. তারকা খেলোয়াড়দের ডেটাবেস (Fantasy 11-এর জন্য)
PLAYERS_DB = [
    {"name": "Shakib Al Hasan", "team": "Bangladesh", "role": "All-Rounder", "style": "spin", "form": 92, "credits": 9.5},
    {"name": "Mustafizur Rahman", "team": "Bangladesh", "role": "Bowler", "style": "pace", "form": 89, "credits": 9.0},
    {"name": "Towhid Hridoy", "team": "Bangladesh", "role": "Batter", "style": "bat", "form": 86, "credits": 8.5},
    {"name": "Litton Das", "team": "Bangladesh", "role": "WK-Batter", "style": "bat", "form": 82, "credits": 8.5},
    {"name": "Rishad Hossain", "team": "Bangladesh", "role": "Bowler", "style": "spin", "form": 88, "credits": 8.0},
    {"name": "Taskin Ahmed", "team": "Bangladesh", "role": "Bowler", "style": "pace", "form": 87, "credits": 8.5},
    {"name": "Virat Kohli", "team": "India", "role": "Batter", "style": "bat", "form": 95, "credits": 10.0},
    {"name": "Jasprit Bumrah", "team": "India", "role": "Bowler", "style": "pace", "form": 96, "credits": 9.5},
    {"name": "Hardik Pandya", "team": "India", "role": "All-Rounder", "style": "pace", "form": 90, "credits": 9.5},
    {"name": "Suryakumar Yadav", "team": "India", "role": "Batter", "style": "bat", "form": 93, "credits": 9.5},
    {"name": "Rishabh Pant", "team": "India", "role": "WK-Batter", "style": "bat", "form": 87, "credits": 9.0},
    {"name": "Kuldeep Yadav", "team": "India", "role": "Bowler", "style": "spin", "form": 91, "credits": 9.0},
    {"name": "Axar Patel", "team": "India", "role": "All-Rounder", "style": "spin", "form": 88, "credits": 8.5},
    {"name": "Tanzim Hasan Sakib", "team": "Bangladesh", "role": "Bowler", "style": "pace", "form": 83, "credits": 8.0},
    {"name": "Mahmudullah", "team": "Bangladesh", "role": "All-Rounder", "style": "spin", "form": 80, "credits": 8.0}
]

# ৩. AI মডেল ট্রেইনিং
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
    
    results = []
    for i in range(n_samples):
        v_bias = VENUES[selected_venues[i]]["chase_bias"]
        death_penalty = -1.5 if (balls_left[i] <= 30 and rrr[i] > 10.5) else 0.0
        win_metric = (wickets_left[i] * 0.48) - (rrr[i] * 0.38) + (crr[i] * 0.12) - (runs_left[i] * 0.015) + (v_bias * 4.5) + death_penalty
        prob = 1 / (1 + np.exp(-win_metric))
        results.append(1 if prob > 0.5 else 0)
        
    df = pd.DataFrame({
        'venue': selected_venues, 'runs_left': runs_left, 'balls_left': balls_left,
        'wickets_left': wickets_left, 'crr': crr, 'rrr': rrr, 'result': results
    })
    
    preprocessor = ColumnTransformer(
        transformers=[('venue_enc', OneHotEncoder(handle_unknown='ignore'), ['venue'])],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42))
    ])
    pipeline.fit(df[['venue', 'runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr']], df['result'])
    return pipeline

model = train_advanced_model()

# ৪. সাইডবার গ্লোবাল কন্ট্রোল
st.sidebar.header("⚙️ গ্লোবাল ম্যাচ সেটিংস")
selected_venue = st.sidebar.selectbox("🏟️ স্টেডিয়াম নির্বাচন করুন:", list(VENUES.keys()))
v_info = VENUES[selected_venue]
st.sidebar.caption(f"গড় রান: **{v_info['avg_score']}** | স্পিন সহায়তা: **{v_info['spin_aid']}%** | পেস সহায়তা: **{v_info['pace_aid']}%**")

# ৫. ৩টি প্রধান ট্যাব তৈরি
tab1, tab2, tab3 = st.tabs([
    "📊 লাইভ ম্যাচ প্রেডিক্টর (Live Win %)", 
    "🪙 টস ও প্রি-ম্যাচ বিশ্লেষণ (Pre-Match AI)", 
    "🌟 AI ফ্যান্টাসি ১১ জেনারেটর (Fantasy Team)"
])

# ==================== TAB 1: LIVE MATCH PREDICTOR ====================
with tab1:
    mode = st.sidebar.radio("ডেটা সোর্স (লাইভ ট্যাবের জন্য):", ["ম্যানুয়াল স্লাইডার (Manual)", "লাইভ ম্যাচ API (CricAPI)"])
    batting_team, bowling_team = "Bangladesh", "India"
    target, current_score, overs_completed, wickets_lost = v_info["avg_score"] + 10, 118, 13.4, 3

    if mode == "লাইভ ম্যাচ API (CricAPI)":
        api_key = st.sidebar.text_input("আপনার CricAPI Key দিন:", type="password")
        if api_key:
            try:
                res = requests.get(f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0").json()
                if res.get("status") == "success":
                    t20s = [m for m in res.get("data", []) if m.get("matchType") == "t20" and m.get("matchStarted")]
                    if t20s:
                        m_name = st.sidebar.selectbox("চলমান T20 ম্যাচ:", [m["name"] for m in t20s])
                        m_data = next(m for m in t20s if m["name"] == m_name)
                        scores, teams = m_data.get("score", []), m_data.get("teams", ["Team A", "Team B"])
                        if len(scores) >= 2:
                            target = scores[0].get("r", 160) + 1
                            current_score = scores[1].get("r", 0)
                            wickets_lost = min(9, scores[1].get("w", 0))
                            overs_completed = float(scores[1].get("o", 1.0))
                            batting_team, bowling_team = teams[1], teams[0]
                    else:
                        st.sidebar.warning("কোনো লাইভ T20 ম্যাচ চলছে না।")
            except Exception as e:
                st.sidebar.error(f"ত্রুটি: {e}")
    else:
        batting_team = st.sidebar.text_input("ব্যাটিং দল", "Bangladesh")
        bowling_team = st.sidebar.text_input("বোলিং দল", "India")
        target = st.sidebar.slider("টার্গেট স্কোর (Target)", 100, 260, 180)
        current_score = st.sidebar.slider("বর্তমান রান (Current Score)", 0, target, 122)
        overs_completed = st.sidebar.slider("সম্পন্ন ওভার (Overs)", 0.1, 19.5, 14.0, 0.1)
        wickets_lost = st.sidebar.slider("পতন হওয়া উইকেট (Wickets)", 0, 9, 3)

    whole_overs = int(overs_completed)
    balls_in_over = min(5, int(round((overs_completed - whole_overs) * 10)))
    balls_played = max(1, (whole_overs * 6) + balls_in_over)
    runs_left = max(0, target - current_score)
    balls_left = max(1, 120 - balls_played)
    wickets_left = 10 - wickets_lost
    crr = round((current_score * 6) / balls_played, 2)
    rrr = round((runs_left * 6) / balls_left, 2)

    if runs_left == 0:
        win_prob, loss_prob = 100.0, 0.0
    else:
        input_df = pd.DataFrame([[selected_venue, runs_left, balls_left, wickets_left, crr, rrr]],
                                columns=['venue', 'runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr'])
        probs = model.predict_proba(input_df)[0]
        loss_prob, win_prob = round(probs[0] * 100, 1), round(probs[1] * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("স্কোরবোর্ড", f"{current_score}/{wickets_lost} ({whole_overs}.{balls_in_over})", f"টার্গেট: {target}")
    col2.metric("সমীকরণ", f"{runs_left} রান দরকার", f"{balls_left} বল বাকি")
    col3.metric("Current Run Rate", f"{crr}", f"পিচ গড়: {v_info['avg_score']}")
    col4.metric("Required Run Rate", f"{rrr}", delta=round(crr - rrr, 2))

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 লাইভ জয়ের সম্ভাব্যতা (Real-Time Win %)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(y=['Win %'], x=[win_prob], name=batting_team, orientation='h', marker=dict(color='#00C853'), text=f"{batting_team}: {win_prob}%", textposition='inside', textfont=dict(size=16, color='white')))
        fig_bar.add_trace(go.Bar(y=['Win %'], x=[loss_prob], name=bowling_team, orientation='h', marker=dict(color='#D50000'), text=f"{bowling_team}: {loss_prob}%", textposition='inside', textfont=dict(size=16, color='white')))
        fig_bar.update_layout(barmode='stack', height=140, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showticklabels=False), yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info(f"🧠 **AI বিশ্লেষণ:** প্রতি ওভারে **{rrr}** রান প্রয়োজন এবং হাতে **{wickets_left}** উইকেট আছে। পিচের আচরণ অনুযায়ী **{batting_team if win_prob >= 50 else bowling_team}** সুবিধাজনক অবস্থানে আছে।")

    with c2:
        pressure_score = min(100, max(0, int((rrr * 5) + ((10 - wickets_left) * 6))))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pressure_score, title={'text': "ব্যাটিং চাপ (Pressure Index)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF6D00"}, 'steps': [{'range': [0, 40], 'color': "#E8F5E9"}, {'range': [40, 70], 'color': "#FFF3E0"}, {'range': [70, 100], 'color': "#FFEBEE"}]}
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 2: PRE-MATCH & TOSS AI ====================
with tab2:
    st.subheader("🪙 টস ও ম্যাচ শুরুর আগের পূর্বাভাস (Pre-Match Toss & Score Predictor)")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        team_a = st.text_input("দল ১ (Team A)", "Bangladesh")
        team_a_strength = st.slider(f"{team_a} টিম রেটিং", 60, 100, 82)
    with p_col2:
        team_b = st.text_input("দল ২ (Team B)", "India")
        team_b_strength = st.slider(f"{team_b} টিম রেটিং", 60, 100, 90)
    with p_col3:
        toss_winner = st.selectbox("টস জয়ী দল:", [team_a, team_b])
        toss_decision = st.selectbox("টস জিতে সিদ্ধান্ত:", ["Fielding (Chase)", "Batting First"])

    # টস ও পিচ বোনাস হিসাব
    toss_bonus = 0
    if toss_decision == "Fielding (Chase)" and v_info["chase_bias"] > 0:
        toss_bonus = 6.5  # শিশির বা চেজিং পিচে ফিল্ডিং নিলে সুবিধা
    elif toss_decision == "Batting First" and v_info["chase_bias"] < 0:
        toss_bonus = 6.5  # স্লো পিচে আগে ব্যাটিং নিলে সুবিধা
    else:
        toss_bonus = -3.0

    a_score = team_a_strength + (toss_bonus if toss_winner == team_a else -toss_bonus)
    b_score = team_b_strength + (toss_bonus if toss_winner == team_b else -toss_bonus)
    
    pre_win_a = round(100 / (1 + np.exp(-(a_score - b_score) / 6)), 1)
    pre_win_b = round(100 - pre_win_a, 1)

    recommended_action = "আগে ফিল্ডিং (Chase)" if v_info["chase_bias"] > 0 else "আগে ব্যাটিং (Defend)"
    proj_low = v_info["avg_score"] - 8
    proj_high = v_info["avg_score"] + 12

    m1, m2, m3 = st.columns(3)
    m1.metric("টস জিতলে AI পরামর্শ", recommended_action)
    m2.metric("১ম ইনিংস সম্ভাব্য স্কোর", f"{proj_low} - {proj_high}")
    m3.metric("জয়ী হওয়ার নিরাপদ স্কোর (Par Score)", f"{v_info['avg_score'] + 10}+")

    st.success(f"🎯 **প্রি-ম্যাচ জয়ের সম্ভাবনা:** **{team_a}: {pre_win_a}%** | **{team_b}: {pre_win_b}%** (টস ও পিচ কন্ডিশন বিবেচনায়)")

# ==================== TAB 3: AI FANTASY 11 GENERATOR ====================
with tab3:
    st.subheader("🌟 AI সেরা ফ্যান্টাসি একাদশ (Dream11 / Fantasy Squad Builder)")
    st.markdown(f"**{selected_venue.split('-')[0]}**-এর পিচে (স্পিন সহায়তা: **{v_info['spin_aid']}%**, পেস সহায়তা: **{v_info['pace_aid']}%**) সর্বোচ্চ পয়েন্ট পাওয়ার সম্ভাব্য একাদশ:")

    fantasy_df = pd.DataFrame(PLAYERS_DB)
    
    # পিচের আচরণ অনুযায়ী ফ্যান্টাসি পয়েন্ট হিসাব
    ai_points = []
    for _, row in fantasy_df.iterrows():
        base = row["form"]
        if row["style"] == "spin":
            bonus = v_info["spin_aid"] * 0.25
        elif row["style"] == "pace":
            bonus = v_info["pace_aid"] * 0.25
        else:
            bonus = (v_info["avg_score"] - 140) * 0.35
        ai_points.append(round(base + bonus, 1))
        
    fantasy_df["AI Projected Points"] = ai_points
    best_11 = fantasy_df.sort_values(by="AI Projected Points", ascending=False).head(11).reset_index(drop=True)
    best_11.index = best_11.index + 1

    captain = best_11.iloc[0]["name"]
    vice_captain = best_11.iloc[1]["name"]

    fc1, fc2 = st.columns(2)
    fc1.metric("👑 সেরা ক্যাপ্টেন (2x Points)", captain, f"সম্ভাব্য পয়েন্ট: {best_11.iloc[0]['AI Projected Points']}")
    fc2.metric("⚡ সেরা ভাইস-ক্যাপ্টেন (1.5x Points)", vice_captain, f"সম্ভাব্য পয়েন্ট: {best_11.iloc[1]['AI Projected Points']}")

    st.dataframe(
        best_11[["name", "team", "role", "credits", "AI Projected Points"]],
        use_container_width=True
        )
