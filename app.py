import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler into the app
model = joblib.load('gaming_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Build the visual interface
st.title("🎮 Player Engagement Predictor")
st.write("Enter a player's stats to predict how engaged they will be.")

# Create input widgets for the user
col1, col2 = st.columns(2)

with col1:
    play_time = st.number_input("Play Time (Hours)", min_value=0.0, max_value=500.0, value=10.0)
    sessions = st.number_input("Sessions Per Week", min_value=0, max_value=50, value=5)

with col2:
    avg_duration = st.number_input("Avg Session Duration (Min)", min_value=0, max_value=300, value=45)
    player_level = st.number_input("Player Level", min_value=1, max_value=100, value=10)

# 3. Create the prediction logic
if st.button("Predict Engagement"):
    
    # Create a DataFrame from the inputs
    # CRITICAL: These column names must match the ones you used during training!
    input_df = pd.DataFrame({
        'PlayTimeHours': [play_time],
        'SessionsPerWeek': [sessions],
        'AvgSessionDurationMinutes': [avg_duration],
        'PlayerLevel': [player_level]
        # Add the rest of your features here...
    })
    
    # Scale the data using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # Make the prediction
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.success(f"Predicted Engagement Level: **{prediction[0]}**")
