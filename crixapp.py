import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# পেজের কনফিগারেশন
st.set_page_config(page_title="AI Cricket Win Predictor", page_icon="🏏", layout="wide")

st.title("🏏 AI T20 Cricket Win Probability Dashboard")
st.markdown("স্লাইডার পরিবর্তন করে ম্যাচের যেকোনো পরিস্থিতির **Real-time Win Probability** এবং গ্রাফ দেখুন।")

# ধাপ ১: সিন্থেটিক ডেটাসেট তৈরি ও মডেল ট্রেইনিং (ক্যাশ করা যাতে বারবার লোড না হয়)
@st.cache_resource
def train_prediction_model():
    np.random.seed(42)
    n_samples = 2000
    
    runs_left = np.random.randint(1, 160, n_samples)
    balls_left = np.random.randint(6, 120, n_samples)
    wickets_left = np.random.randint(1, 11, n_samples)
    crr = np.random.uniform(5.0, 12.0, n_samples)
    rrr = (runs_left * 6) / balls_left
    
    # জয়ের লজিক্যাল সমীকরণ (উইকেট বেশি ও RRR কম থাকলে জয়ের সম্ভাবনা বেশি)
    score_metric = (wickets_left * 4.5) - (rrr * 3.2) + (crr * 1.2) - (runs_left * 0.12)
    prob = 1 / (1 + np.exp(-score_metric / 5))
    result = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'crr': crr,
        'rrr': rrr,
        'result': result
    })
    
    X = df[['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr']]
    y = df['result']
    
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_prediction_model()

# ধাপ ২: সাইডবারে ইউজার ইনপুট ও স্লাইডার
st.sidebar.header("⚙️ ম্যাচের বর্তমান অবস্থা দিন")

batting_team = st.sidebar.text_input("ব্যাটিং দল", "Bangladesh")
bowling_team = st.sidebar.text_input("বোলিং দল", "India")

target = st.sidebar.slider("টার্গেট স্কোর (Target)", min_value=100, max_value=260, value=185, step=1)
current_score = st.sidebar.slider("বর্তমান রান (Current Score)", min_value=0, max_value=target, value=128, step=1)
overs_completed = st.sidebar.slider("সম্পন্ন ওভার (Overs)", min_value=1.0, max_value=19.5, value=14.0, step=0.5)
wickets_lost = st.sidebar.slider("পতন হওয়া উইকেট (Wickets)", min_value=0, max_value=9, value=3, step=1)

# ধাপ ৩: গাণিতিক হিসাব (Feature Engineering)
whole_overs = int(overs_completed)
balls_in_over = int(round((overs_completed - whole_overs) * 6))
balls_played = (whole_overs * 6) + balls_in_over

runs_left = max(0, target - current_score)
balls_left = max(1, 120 - balls_played)
wickets_left = 10 - wickets_lost

crr = round((current_score * 6) / balls_played, 2) if balls_played > 0 else 0.0
rrr = round((runs_left * 6) / balls_left, 2) if balls_left > 0 else 0.0

# ম্যাচ জয়ের কন্ডিশন চেক ও প্রেডিকশন
if runs_left == 0:
    win_prob, loss_prob = 100.0, 0.0
else:
    input_df = pd.DataFrame([[runs_left, balls_left, wickets_left, crr, rrr]],
                            columns=['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr'])
    probs = model.predict_proba(input_df)[0]
    loss_prob = round(probs[0] * 100, 1)
    win_prob = round(probs[1] * 100, 1)

# ধাপ ৪: মেইন ড্যাশবোর্ড মেট্রিক্স (KPI Cards)
col1, col2, col3, col4 = st.columns(4)
col1.metric("স্কোরবোর্ড", f"{current_score}/{wickets_lost}", f"টার্গেট: {target}")
col2.metric("জয়ের জন্য প্রয়োজন", f"{runs_left} রান", f"বাকি {balls_left} বলে")
col3.metric("Current Run Rate (CRR)", f"{crr}")
col4.metric("Required Run Rate (RRR)", f"{rrr}", delta=round(crr - rrr, 2))

st.divider()

# ধাপ ৫: গ্রাফ ১ - Win Probability Horizontal Bar (ALLCRIC স্টাইল)
st.subheader("📊 বর্তমান জয়ের সম্ভাব্যতা (Real-Time Win Probability)")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=['Win Probability'], x=[win_prob], name=batting_team,
    orientation='h', marker=dict(color='#00C853'),
    text=f"{batting_team}: {win_prob}%", textposition='inside', insidetextanchor='middle',
    textfont=dict(size=16, color='white')
))
fig_bar.add_trace(go.Bar(
    y=['Win Probability'], x=[loss_prob], name=bowling_team,
    orientation='h', marker=dict(color='#FF3D00'),
    text=f"{bowling_team}: {loss_prob}%", textposition='inside', insidetextanchor='middle',
    textfont=dict(size=16, color='white')
))

fig_bar.update_layout(
    barmode='stack', height=130, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
    yaxis=dict(showticklabels=False),
    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_bar, use_container_width=True)

# ধাপ ৬: গ্রাফ ২ - ভবিষ্যৎ ওভারগুলোতে RRR ও জয়ের সম্ভাবনার প্রজেকশন গ্রাফ
st.subheader("📈 পরবর্তী ওভারগুলোতে জয়ের সম্ভাবনার প্রজেকশন")
st.caption("যদি বর্তমান রান রেট (CRR) বজায় থাকে এবং আর কোনো উইকেট না পড়ে, তবে প্রতি ওভারে জয়ের সম্ভাবনা যেভাবে বদলাবে:")

future_overs = list(range(whole_overs, 20))
projected_probs = []

for ov in future_overs:
    b_played = ov * 6
    b_left = max(1, 120 - b_played)
    proj_score = min(target, int((crr / 6) * b_played))
    r_left = max(0, target - proj_score)
    proj_rrr = (r_left * 6) / b_left
    
    if r_left == 0:
        projected_probs.append(100.0)
    else:
        temp_df = pd.DataFrame([[r_left, b_left, wickets_left, crr, proj_rrr]],
                               columns=['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr'])
        p = model.predict_proba(temp_df)[0][1] * 100
        projected_probs.append(round(p, 1))

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=future_overs, y=projected_probs, mode='lines+markers',
    name=f'{batting_team} Win %', line=dict(color='#00E676', width=3),
    fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.15)'
))
fig_line.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% সমতা রেখা")
fig_line.update_layout(
    xaxis_title="ওভার (Overs)", yaxis_title="জয়ের সম্ভাবনা (%)",
    yaxis=dict(range=[0, 105]), height=360, margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig_line, use_container_width=True)