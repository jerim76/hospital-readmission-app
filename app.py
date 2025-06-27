# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Page configuration
st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide")
st.title("🏥 Hospital Readmission Risk Prediction")
st.markdown("Predict 30-day readmission risk for discharged patients")

# Sidebar for user input
with st.sidebar:
    st.header("Patient Information")
    
    # Demographic information
    age = st.slider("Age", 18, 100, 60)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
    diagnosis = st.selectbox("Primary Diagnosis", ["Diabetes", "Heart Disease", "Infection", "Injury"])
    a1c_result = st.selectbox("A1C Result", ["Normal", "Abnormal", "Not Tested"])
    
    # Medical history
    st.subheader("Medical History")
    num_lab_procedures = st.slider("Lab Procedures", 0, 100, 40)
    num_medications = st.slider("Medications", 0, 50, 15)
    num_outpatient_visits = st.slider("Outpatient Visits (past year)", 0, 20, 3)
    num_inpatient_visits = st.slider("Inpatient Visits (past year)", 0, 10, 2)
    num_emergency_visits = st.slider("Emergency Visits (past year)", 0, 10, 1)
    num_diagnoses = st.slider("Number of Diagnoses", 1, 15, 5)
    
    # Prediction button
    predict_btn = st.button("Predict Readmission Risk")
    train_btn = st.button("Train Model")

# Feature engineering function
def create_features(df):
    # Clinical complexity
    df['Complex_Case_Flag'] = ((df['Num_Diagnoses'] > 5) & (df['Num_Medications'] > 10)).astype(int)
    
    # Care utilization
    df['Medication_Intensity'] = df['Num_Medications'] / (df['Num_Diagnoses'] + 1e-5)
    df['Total_Visits'] = df[['Num_Outpatient_Visits', 'Num_Inpatient_Visits', 'Num_Emergency_Visits']].sum(axis=1)
    df['Recent_Care_Intensity'] = (
        df['Num_Emergency_Visits'] * 0.6 + 
        df['Num_Inpatient_Visits'] * 0.3 +
        df['Num_Outpatient_Visits'] * 0.1
    )
    
    # Chronic conditions
    chronic_conditions = ['Diabetes', 'Heart Disease']
    df['Chronic_Condition'] = df['Diagnosis'].apply(
        lambda x: 1 if any(cond in x for cond in chronic_conditions) else 0
    )
    return df

# Function to train the model
def train_model():
    # Generate synthetic hospital data
    def generate_hospital_data(n_samples=1000):
        np.random.seed(42)
        patient_ids = np.arange(1, n_samples + 1)
        ages = np.clip(np.random.normal(60, 20, n_samples), 18, 100)
        genders = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.1])
        admission_types = np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples, p=[0.4, 0.4, 0.2])
        diagnoses = np.random.choice(['Diabetes', 'Heart Disease', 'Infection', 'Injury'], n_samples, p=[0.3, 0.3, 0.2, 0.2])
        num_lab_procedures = np.random.poisson(40, n_samples)
        num_medications = np.random.poisson(15, n_samples)
        num_outpatient_visits = np.random.poisson(3, n_samples)
        num_inpatient_visits = np.random.poisson(2, n_samples)
        num_emergency_visits = np.random.poisson(1, n_samples)
        num_diagnoses = np.random.randint(1, 10, n_samples)
        a1c_results = np.random.choice(['Normal', 'Abnormal', np.nan], n_samples, p=[0.5, 0.4, 0.1])
        
        # Create readmission target
        readmission_risk = (
            0.1 * (ages > 65) +
            0.2 * (num_medications > 15) +
            0.3 * (num_diagnoses > 5) +
            0.15 * (a1c_results == 'Abnormal') +
            0.25 * (diagnoses == 'Diabetes') +
            0.1 * (num_emergency_visits > 2) -
            0.1 * (admission_types == 'Elective') +
            np.random.normal(0, 0.2, n_samples)
        )
        readmitted = (readmission_risk > 0.5).astype(int)
        readmitted_labels = np.where(readmitted == 1, 'Yes', 'No')
        
        return pd.DataFrame({
            'Patient_ID': patient_ids,
            'Age': ages,
            'Gender': genders,
            'Admission_Type': admission_types,
            'Diagnosis': diagnoses,
            'Num_Lab_Procedures': num_lab_procedures,
            'Num_Medications': num_medications,
            'Num_Outpatient_Visits': num_outpatient_visits,
            'Num_Inpatient_Visits': num_inpatient_visits,
            'Num_Emergency_Visits': num_emergency_visits,
            'Num_Diagnoses': num_diagnoses,
            'A1C_Result': a1c_results,
            'Readmitted': readmitted_labels
        })
    
    # Generate and prepare data
    with st.spinner("Generating training data..."):
        df = generate_hospital_data(1500)
        df = create_features(df)
    
    # Feature definitions
    numerical_features = [
        'Age', 'Num_Lab_Procedures', 'Num_Medications', 
        'Num_Outpatient_Visits', 'Num_Inpatient_Visits',
        'Num_Emergency_Visits', 'Num_Diagnoses',
        'Medication_Intensity', 'Total_Visits', 'Recent_Care_Intensity'
    ]
    
    categorical_features = [
        'Gender', 'Admission_Type', 'Diagnosis', 'A1C_Result',
        'Complex_Case_Flag', 'Chronic_Condition'
    ]
    
    # Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare target and features
    y = df['Readmitted'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Readmitted', 'Patient_ID'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Calculate class weights
    readmit_ratio = y_train.value_counts(normalize=True)
    scale_pos_weight = readmit_ratio[0] / readmit_ratio[1]
    
    # Build and train model
    with st.spinner("Training model..."):
        model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
            ('classifier', XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.trained = True
        
        # Evaluate model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        st.success("Model trained successfully!")
        return report, cm

# Train model if requested
if train_btn:
    report, cm = train_model()
    st.session_state.report = report
    st.session_state.cm = cm

# Main content area
if predict_btn:
    if not st.session_state.trained:
        st.warning("Please train the model first!")
        st.stop()
    
    # Create patient dataframe
    patient_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Admission_Type': admission_type,
        'Diagnosis': diagnosis,
        'Num_Lab_Procedures': num_lab_procedures,
        'Num_Medications': num_medications,
        'Num_Outpatient_Visits': num_outpatient_visits,
        'Num_Inpatient_Visits': num_inpatient_visits,
        'Num_Emergency_Visits': num_emergency_visits,
        'Num_Diagnoses': num_diagnoses,
        'A1C_Result': a1c_result
    }])
    
    # Apply feature engineering
    patient_data = create_features(patient_data)
    
    # Make prediction
    model = st.session_state.model
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"🚨 High readmission risk: {probability:.1%}")
    else:
        st.success(f"✅ Low readmission risk: {1-probability:.1%}")
    
    # Create progress bar for risk score
    st.progress(probability)
    st.caption(f"Readmission probability: {probability:.1%}")
    
    # Display SHAP explanation
    st.subheader("Risk Factor Analysis")
    
    try:
        # Process patient data
        preprocessor = model.named_steps['preprocessor']
        processed_data = preprocessor.transform(patient_data)
        
        # Get feature names
        num_features = numerical_features
        cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        all_features = list(num_features) + list(cat_features)
        
        # Generate SHAP values
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(processed_data)
        
        # Plot waterfall
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            feature_names=all_features
        ), show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not generate explanation: {str(e)}")

# Display model information if trained
if st.session_state.trained:
    st.sidebar.success("Model is trained and ready!")
    
    if 'report' in st.session_state:
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Precision", f"{st.session_state.report['1']['precision']:.1%}")
            st.metric("Recall", f"{st.session_state.report['1']['recall']:.1%}")
        
        with col2:
            st.metric("F1-Score", f"{st.session_state.report['1']['f1-score']:.1%}")
            st.metric("Accuracy", f"{st.session_state.report['accuracy']:.1%}")
        
        # Confusion matrix
        fig, ax = plt.subplots()
        ax.matshow(st.session_state.cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(st.session_state.cm.shape[0]):
            for j in range(st.session_state.cm.shape[1]):
                ax.text(x=j, y=i, s=st.session_state.cm[i, j], 
                        va='center', ha='center', size='large')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Low Risk', 'High Risk'])
        ax.set_yticklabels(['Low Risk', 'High Risk'])
        st.pyplot(fig)
else:
    st.info("Click 'Train Model' in the sidebar to start")
    st.image("https://images.unsplash.com/photo-1581595219319-5b467da3d8d8?auto=format&fit=crop&w=1200&h=400", 
             caption="Hospital Readmission Prediction System")

# Add footer
st.markdown("---")
st.caption("Hospital Readmission Prediction System v1.0 | AI for Healthcare")
