import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load pipeline
pipeline = joblib.load("model_features.pkl")
model = pipeline['model']
scaler = pipeline['scaler']
selector = pipeline['selector']

# Feature list (used in training)
features = ['protein', 'fat', 'carbohydrate', 'fiber']

# Load original dataset for recommendation
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/RAJDEEP/Downloads/nutrition.csv")
    
    def clean_numeric_safe(val):
        if isinstance(val, str):
            cleaned = ''.join(ch for ch in val if ch.isdigit() or ch == '.' or ch == '-')
            try:
                return float(cleaned)
            except:
                return np.nan
        return val

    for col in df.columns:
        if col not in ['Unnamed: 0', 'name', 'serving_size']:
            df[col] = df[col].apply(clean_numeric_safe)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df_orig = load_data()

# App title
st.title("🍽️ Nutrition Analysis & Food Recommendation App")

# ----------------------------
# Section 1: Predict Calories
# ----------------------------
st.header("🔢 Predict Calories from Nutrients")

input_data = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0, min_value=0.0)

if st.button("Predict Calories"):
    if any(value < 0 for value in input_data.values()):
        st.error("❌ Please enter valid (non-negative) values for all nutrients.")
    else:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)
        st.success(f"🔍 Estimated Calories: **{prediction[0]:.2f} kcal**")

# ----------------------------
# Section 2: Recommend Foods
# ----------------------------
st.header("🎯 Recommend Foods Based on Nutrition Goals")

max_calories = st.slider("Maximum Calories", 50, 1000, 200)
min_protein = st.slider("Minimum Protein (g)", 0, 50, 5)
max_fat = st.slider("Maximum Fat (g)", 0, 50, 10)
top_n = st.slider("Top N Foods", 1, 50, 10)

if st.button("Recommend Foods"):
    df_rec = df_orig[
        (df_orig['calories'] <= max_calories) &
        (df_orig['protein'] >= min_protein) &
        (df_orig['fat'] <= max_fat)
    ]
    result = df_rec[['name', 'calories', 'protein', 'fat', 'carbohydrate']].sort_values(by='protein', ascending=False).head(top_n)
    
    if result.empty:
        st.warning("⚠️ No matching foods found. Try adjusting the filters.")
    else:
        st.dataframe(result.reset_index(drop=True))

# ----------------------------
# Section 3: Visualize Nutrients
# ----------------------------
st.header("📊 Nutrient Breakdown for Selected Food")

food_options = df_orig['name'].dropna().unique()
selected_food = st.selectbox("Choose a food item", sorted(food_options))

if selected_food:
    item = df_orig[df_orig['name'] == selected_food].iloc[0]
    nutrients = {
        'Protein (g)': item['protein'],
        'Fat (g)': item['fat'],
        'Carbohydrates (g)': item['carbohydrate']
    }

    # Calories from macros
    cals_from = {
        'Protein': item['protein'] * 4,
        'Fat': item['fat'] * 9,
        'Carbohydrates': item['carbohydrate'] * 4
    }

    # --- Bar Chart ---
    st.subheader("Macronutrient Composition (g)")
    st.bar_chart(pd.DataFrame([nutrients], index=[selected_food]))

    # --- Pie Chart ---
    st.subheader("Calorie Contribution by Macronutrients")
    fig, ax = plt.subplots()
    ax.pie(cals_from.values(), labels=cals_from.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Footer
st.markdown("---")
