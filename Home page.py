import streamlit as st
import pickle
import pandas as pd

def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_data(data):
    df = pd.DataFrame([data])
    df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})
    df['age_category'] = df['age'].apply(categorize_age)
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    df['children_str'] = df['children'].apply(lambda x: str(x))
    df = pd.get_dummies(df, columns=['age_category', 'children_str', 'bmi_category']).astype(int)
    print(df)

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'under_weight'
    elif 18.5 <= bmi < 25:
        return 'normal_weight'
    elif 25 <= bmi < 30:
        return 'over_weight'
    else:
        return 'obese'

def categorize_age(age):
    if 18 < age < 26:
        return 'young_adult'
    elif 26 <= age < 36:
        return 'early_adulthood'
    elif 36 <= age < 46:
        return 'mid_adulthood'
    else:
        return 'late_adulthood'

def main():
    st.title("Real-Time Prediction Application")

    st.header("Enter User Information:")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    children = st.number_input("Number of Children", min_value=0, max_value=20, step=1)
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])


    # Create data structure for prediction
    input_data = {
        "age": age,
        "bmi": weight / ((height / 100) ** 2),
        "smoker": smoker,
        "children_str_0" : 1 if children == "0" else 0,
        "bmi_category_obese" : 1 if (weight / ((height / 100) ** 2)) > 30 else 0,
        "age_category_early_adulthood" : 1 if 26 <= age < 36 else 0,
        'bmi_category_over_weight': 1 if 25 <= (weight / ((height / 100) ** 2)) < 30 else 0
    }

    # Load the model
    model = load_model()

    # Predict if data is complete
    if st.button("Generate Prediction"):
        preprocess_data(input_data)
        preprocessed_data = preprocess_data(input_data)
        prediction = model.predict(preprocessed_data)
        st.write(f"Hello {first_name} {last_name}, here is the prediction:")
        st.success(f"Result: {prediction[0]}")

if __name__ == "__main__":
    main()