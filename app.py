import streamlit as st
import pandas as pd
import numpy as np
import joblib 
# Load the trained machine learning model
model = joblib.load('employee_retention_model.pkl')

# Load preprocessing information
preprocessing = joblib.load('preprocessing_components.pkl')

# Load feature column names
feature_columns = joblib.load('model_feature_columns.pkl')

# Set the application title

st.set_page_config(page_title="Employee Retention Prediction")
st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

h1 {
    color: #16324F;
    font-weight: 700;
}

h2, h3 {
    color: #234E70;
}

.stButton > button {
    background-color: #16324F;
    color: white !important;
    border: 1px solid #16324F;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #234E70 !important;
    color: white !important;
    border: 1px solid #234E70;
}

.stButton > button:focus {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#16324F; padding:25px; border-radius:12px; margin-bottom:25px;">
    <h1 style="color:white; text-align:center;">
        HR Employee Retention Analytics
    </h1>
    <p style="color:#DCE6F1; text-align:center; font-size:16px;">
        AI-Powered Employee Job Change Prediction System
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Analyze employee information, identify potential job-change risks, "
    "and support data-driven HR retention decisions."
)

# Create the employee information section

st.header("Employee Information")

st.write("Enter the employee details below to predict job change possibility.")
# Enter City Development Index

city_development_index = st.number_input(
    "City Development Index",
    min_value=0.0,
    max_value=1.0,
    value=0.90
)
# Select employee gender

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other", "Unknown"]
)
# Select relevant work experience

relevent_experience = st.selectbox(
    "Relevant Experience",
    ["Has relevent experience", "No relevent experience"]
)
# Select university enrolment status

enrolled_university = st.selectbox(
    "Enrolled University",
    ["Full time course", "Part time course", "no_enrollment", "Unknown"]
)
# Select employee education level

education_level = st.selectbox(
    "Education Level",
    ["Graduate", "High School", "Masters", "Phd", "Primary School", "Unknown"]
)
# Select employee major discipline

major_discipline = st.selectbox(
    "Major Discipline",
    ["STEM", "Humanities", "Business Degree", "Arts", "No Major", "Other", "Unknown"]
)
# Select total work experience

experience = st.selectbox(
    "Work Experience (Years)",
    ["<1"] + [str(i) for i in range(1, 21)] + [">20", "Unknown"]
)
# Select current company size

company_size = st.selectbox(
    "Company Size",
    ["<10", "10/49", "50-99", "100-500", "500-999",
     "1000-4999", "5000-9999", "10000+", "Unknown"]
)
# Select company type

company_type = st.selectbox(
    "Company Type",
    ["Early Stage Startup", "Funded Startup", "NGO",
     "Other", "Public Sector", "Pvt Ltd", "Unknown"]
)
# Select when the employee last changed jobs

last_new_job = st.selectbox(
    "Last New Job",
    ["never", "1", "2", "3", "4", ">4", "Unknown"]
)
# Enter employee training hours

training_hours = st.number_input(
    "Training Hours",
    min_value=0,
    max_value=500,
    value=40
)
# Select employee city

city = st.selectbox(
    "City",
    sorted(preprocessing['city_frequency_mapping'].keys())
)
# Create prediction button

predict_button = st.button("Predict Job Change")
# Collect all employee details into a DataFrame

if predict_button:

    input_data = pd.DataFrame({
        'city': [city],
        'city_development_index': [city_development_index],
        'gender': [gender],
        'relevent_experience': [relevent_experience],
        'enrolled_university': [enrolled_university],
        'education_level': [education_level],
        'major_discipline': [major_discipline],
        'experience': [experience],
        'company_size': [company_size],
        'company_type': [company_type],
        'last_new_job': [last_new_job],
        'training_hours': [training_hours]
    })

    st.write("Employee Details:")
    st.dataframe(input_data)

    # Create a copy of employee data
    processed_data = input_data.copy()

    # Fill missing categorical values
    categorical_features = [
        'city', 'gender', 'relevent_experience',
        'enrolled_university', 'education_level',
        'major_discipline', 'experience',
        'company_size', 'company_type', 'last_new_job'
    ]

    processed_data[categorical_features] = (
        processed_data[categorical_features].fillna('Unknown')
    )

    # Handle missing numerical values
    numerical_features = [
        'city_development_index', 'training_hours'
    ]

    processed_data[numerical_features] = (
        preprocessing['numerical_imputer'].transform(
            processed_data[numerical_features]
        )
    )
    # Cap extreme training hours using the training data limit

    processed_data['training_hours'] = processed_data['training_hours'].clip(
        upper=preprocessing['training_hours_upper_limit']
    )
    # Keep capped raw hours for training intensity feature

    processed_data['training_hours_raw'] = processed_data['training_hours']

    # Apply log transformation to training hours

    processed_data['training_hours'] = np.log1p(
        processed_data['training_hours']
    )
    # Convert experience into numerical values

    processed_data['experience'] = (
        processed_data['experience'].map(
            preprocessing['experience_mapping']
        )
    )

    # Convert company size into numerical values

    processed_data['company_size'] = (
        processed_data['company_size'].map(
            preprocessing['company_size_mapping']
        )
    )

    # Convert last new job into numerical values

    processed_data['last_new_job'] = (
        processed_data['last_new_job'].map(
            preprocessing['last_new_job_mapping']
        )
    )
    # Create experience categories

    processed_data['experience_category'] = pd.cut(
        processed_data['experience'],
        bins=[0, 1, 10, 21],
        labels=['Fresher', 'Mid-level', 'Senior'],
        include_lowest=True
    ).astype(str)

    processed_data['experience_category'] = (
        processed_data['experience_category'].replace('nan', 'Unknown')
    )
    # Create training intensity categories

    processed_data['training_intensity'] = pd.cut(
        processed_data['training_hours_raw'],
        bins=[-1, 40, 100, float('inf')],
        labels=['Low', 'Medium', 'High']
    ).astype(str)
    # Identify employees who have never changed jobs

    processed_data['never_changed_job'] = (
        processed_data['last_new_job'] == 0
    ).astype(int)

    # Identify early-career employees who have never changed jobs

    processed_data['early_career_never_changed'] = (
        (processed_data['experience_category'] == 'Fresher') &
        (processed_data['never_changed_job'] == 1)
    ).astype(int)
    # Convert city into frequency values

    processed_data['city'] = (
        processed_data['city'].map(
            preprocessing['city_frequency_mapping']
        ).fillna(0)
    )
    # Select numerical features for scaling

    scaling_features = [
        'city',
        'city_development_index',
        'experience',
        'company_size',
        'last_new_job',
        'training_hours'
    ]

    # Apply the saved scaler

    processed_data[scaling_features] = (
        preprocessing['scaler'].transform(
            processed_data[scaling_features]
        )
    )
    # Convert categorical features into dummy variables

    processed_data = pd.get_dummies(
        processed_data,
        columns=[
            'gender',
            'relevent_experience',
            'enrolled_university',
            'education_level',
            'major_discipline',
            'company_type',
            'experience_category',
            'training_intensity'
        ],
        dtype=int
    )
    # Match the exact feature columns used during model training

    processed_data = processed_data.reindex(
        columns=feature_columns,
        fill_value=0
    )
    # Generate employee job change prediction

    prediction = model.predict(processed_data)[0]

    # Calculate probability of job change

    prediction_probability = model.predict_proba(processed_data)[0][1]
    # Display prediction results

    st.subheader("Prediction Result")

    st.write("Probability of Job Change:")

    st.progress(float(prediction_probability))

    st.write(
        f"{prediction_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.warning(
            "The employee may have a higher possibility of looking for a job change."
        )
    else:
        st.success(
            "The employee is less likely to look for a job change."
        )
        # Provide HR recommendations

    st.subheader("HR Recommendation")

    if prediction == 1:
        st.write(
            "HR should consider a retention discussion with the employee, "
            "understand their career expectations, review growth opportunities, "
            "and identify possible reasons for dissatisfaction."
        )
    else:
        st.write(
            "HR can continue regular employee engagement activities, "
            "career development discussions, and periodic feedback sessions."
        )




