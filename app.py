import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the model
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'pipe.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title('🏏 IPL Win Probability Predictor')

# Team Selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

# City Selection
selected_city = st.selectbox('Select Host City', sorted(cities))

# Target Runs
target = st.number_input('Target Score', min_value=1, step=1)

# Match Progress Details
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
with col5:
    wickets_fallen = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1)

# Predict Button
if st.button('Predict Probability'):
    # Validations
    if overs == 0:
        st.error("Overs cannot be zero.")
    elif score > target:
        st.error("Score cannot be greater than target.")
    else:
        # Calculate features
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_fallen
        crr = score / overs if overs > 0 else 0  # Avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

        # Create DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict Probabilities
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display Results
        st.subheader(f"🏏 {batting_team} Win Probability: **{round(win_prob * 100)}%**")
        st.subheader(f"🏏 {bowling_team} Win Probability: **{round(loss_prob * 100)}%**")
