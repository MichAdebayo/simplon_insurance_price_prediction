import streamlit as st
import pickle
import pandas as pd


def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def preprocess_data(data):
    expected_columns = [
        "smoker",
        "age",
        "bmi",
        "age_category_young_adult",
        "age_category_early_adulthood",
        "bmi_category_over_weight",
        "bmi_category_obese",
        "children_str_0",
    ]
    df = pd.DataFrame([data])
    df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})
    df["age_category"] = df["age"].apply(categorize_age)
    df["bmi_category"] = df["bmi"].apply(categorize_bmi)
    df["children_str"] = df["children"].apply(lambda x: str(int(x)))
    df = pd.get_dummies(df, columns=["age_category", "bmi_category", "children_str"])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    return df


def categorize_bmi(bmi):
    if bmi < 18.5:
        return "under_weight"
    elif 18.5 <= bmi < 25:
        return "normal_weight"
    elif 25 <= bmi < 30:
        return "over_weight"
    else:
        return "obese"


def categorize_age(age):
    if 18 < age < 26:
        return "young_adult"
    elif 26 <= age < 36:
        return "early_adulthood"
    elif 36 <= age < 46:
        return "mid_adulthood"
    else:
        return "late_adulthood"


def main():
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.5em; color: #4CAF50; text-align: center; font-weight: bold; }
        .sub-title { font-size: 1.2em; color: #555; text-align: center; margin-bottom: 30px; }
        </style>
        <div class="main-title">Real-Time Prediction Application</div>
        <div class="sub-title">Your trusted tool for insurance charges prediction</div>
        """,
        unsafe_allow_html=True,
    )

    if "selected_sex" not in st.session_state:
        st.session_state.selected_sex = None
    if "selected_smoker" not in st.session_state:
        st.session_state.selected_smoker = None
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = None

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")

    st.write("Gender")
    column1, column2, column3, column4, column5 = st.columns(5)
    with column1:
        if st.button("Male", key="male"):
            st.session_state.selected_sex = "Male"
    with column2:
        if st.button("Female", key="female"):
            st.session_state.selected_sex = "Female"
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    st.write("Smoker")
    column1, column2, column3, column4, column5 = st.columns(5)
    with column1:
        if st.button("Yes", key="smoker_yes"):
            st.session_state.selected_smoker = "Yes"
    with column2:
        if st.button("No", key="smoker_no"):
            st.session_state.selected_smoker = "No"
    height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)

    st.write("Region")
    column1, column2, column3, column4, column5 = st.columns(5)
    with column1:
        if st.button("Northeast", key="northeast"):
            st.session_state.selected_region = "Northeast"
    with column2:
        if st.button("Northwest", key="northwest"):
            st.session_state.selected_region = "Northwest"
    with column3:
        if st.button("Southeast", key="southeast"):
            st.session_state.selected_region = "Southeast"
    with column4:
        if st.button("Southwest", key="southwest"):
            st.session_state.selected_region = "Southwest"
    children = st.number_input("Number of Children", min_value=0, max_value=20, step=1)
    column1, column2, column3, column4, column5 = st.columns(5)
    with column5:
        submit_button = st.button(label="Generate Prediction")

    if submit_button:
        if not st.session_state.selected_sex:
            st.warning("Please select a gender!")
        elif not st.session_state.selected_smoker:
            st.warning("Please select smoker status!")
        elif not st.session_state.selected_region:
            st.warning("Please select a region!")
        else:
            bmi = weight / ((height / 100) ** 2)
            input_data = {
                "age": age,
                "bmi": bmi,
                "smoker": st.session_state.selected_smoker,
                "children": children,
            }

            model = load_model()
            preprocessed_data = preprocess_data(input_data)
            prediction = model.predict(preprocessed_data)
            st.write(f"Hello {first_name} {last_name}, here is the prediction:")
            st.success(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
