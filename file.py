import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from google import genai

# =====================================================
# LOAD ML MODEL & ENCODERS
# =====================================================
model = joblib.load("mood_model.pkl")
encoders = joblib.load("encoders.pkl")

# =====================================================
# GEMINI CONFIG (ONLY REQUESTED MODEL)
# =====================================================
client = genai.Client(
    api_key="AIzaSyDR2cZVTLhb1RDFJn7JKdrykY6-YDPEgJA"   # 🔐 move to st.secrets for hosting
)

MODEL_NAME = "models/gemini-2.5-flash-lite"

# =====================================================
# MOOD → FOOD MAP
# =====================================================
mood_foods = {
    "Sad": ["Banana", "Dark Chocolate", "Salmon", "Berries"],
    "Angry": ["Green Tea", "Yogurt", "Oatmeal"],
    "Joy": ["Fruits", "Smoothies", "Avocado"],
    "Smile": ["Eggs", "Spinach", "Sweet Potato"]
}

# =====================================================
# PROMPT BUILDER
# =====================================================
def generate_prompt(name, mood, water, sleep, exercise, fav_food, suggestions):
    return f"""
User Name: {name}
Predicted Mood: {mood}

Lifestyle Details:
- Sleep: {sleep} hours
- Water Intake: {water} liters
- Exercise: {exercise} hours
- Favourite Food: {fav_food}

Instructions:
1. Briefly explain the user's current mood
2. Suggest foods that help improve this mood
3. Give personalized advice on sleep, water, and exercise
4. Keep the tone friendly, motivating, and positive

Additional Suggestions:
{suggestions}
"""

# =====================================================
# GEMINI CALL (RETRY FOR 503 OVERLOAD)
# =====================================================
def call_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text

        except Exception as e:
            error = str(e).lower()

            if "503" in error or "overloaded" in error:
                time.sleep(2)
                if attempt == retries - 1:
                    return "⚠️ AI server is busy. Please try again in a moment."
            else:
                return f"⚠️ Gemini Error: {e}"

    return "⚠️ AI service temporarily unavailable."

# =====================================================
# STREAMLIT APP
# =====================================================
def main():
    st.title("🧠 NEURODIET: AI-Powered Mood & Nutrition Assistant")
    st.markdown("Personalized wellness advice based on lifestyle habits and AI mood prediction.")

    # ================= USER INPUTS =================
    name = st.text_input("Enter your name")
    age = st.number_input("Age", 10, 100)

    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    menstrual = st.selectbox("Menstrual Cycle Status", encoders["Menstrual"].classes_)

    sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
    water = st.slider("Water Intake (Liters)", 0.0, 5.0, 2.0)
    steps = st.number_input("Steps Walked Today", 0, 20000)
    exercise = st.slider("Exercise Hours", 0.0, 5.0, 0.5)

    behaviour = st.selectbox("Behaviour", encoders["Behaviour"].classes_)
    character = st.selectbox("Character", encoders["Character"].classes_)
    fav_food = st.selectbox("Favourite Food", encoders["Favourite_Food"].classes_)

    # ================= ACTION =================
    if st.button("🎯 Predict Mood & Get Advice"):
        features = [
            age,
            encoders["Gender"].transform([gender])[0],
            encoders["Menstrual"].transform([menstrual])[0],
            sleep,
            water,
            steps,
            exercise,
            encoders["Behaviour"].transform([behaviour])[0],
            encoders["Character"].transform([character])[0],
            encoders["Favourite_Food"].transform([fav_food])[0]
        ]

        # ===== Predict Mood =====
        prediction = model.predict([features])[0]
        mood = encoders["Mood"].inverse_transform([prediction])[0]

        # ===== Rule-Based Suggestions =====
        suggestions = []
        if water < 2:
            suggestions.append("• Increase daily water intake to at least 2 liters.")
        if sleep < 6:
            suggestions.append("• Aim for 7–8 hours of quality sleep.")
        if exercise < 0.5:
            suggestions.append("• Add at least 30 minutes of physical activity.")

        mood_food = mood_foods.get(mood, ["Fruits", "Vegetables"])
        if fav_food in mood_food:
            suggestions.append(f"• Your favourite food '{fav_food}' supports your mood.")
        else:
            suggestions.append(f"• Mood-boosting foods: {', '.join(mood_food)}")

        # ===== Gemini AI Advice =====
        prompt = generate_prompt(
            name, mood, water, sleep, exercise, fav_food,
            "\n".join(suggestions)
        )

        with st.spinner("🤖 Generating personalized advice..."):
            advice = call_gemini(prompt)

        # ===== OUTPUT =====
        st.subheader(f"🧠 Predicted Mood: `{mood}`")
        st.markdown("---")
        st.markdown(advice)

# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    main()
