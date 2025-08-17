import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os

# Set page config
st.set_page_config(page_title="CRS Math Profs Predictor", layout="wide")

# Title
st.title("CRS Math Professors Predictor and Dashboard")

# File paths
DATA_PATH = r"CRS Math Profs (2018-2024).csv"
MODEL_PATH = r"best_rf_model_compressed.pkl"

# Check if files exist
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# Load data and model
@st.cache_data
def load_data():
    # Load the actual data with proper data types
    df = pd.read_csv(DATA_PATH, dtype={"Number": object})
    
    # Select relevant columns
    df = df[['Number', 'Day', 'Room', 'Prof', 'Year', 'Semester', 'Start_time', 'End_time']]
    
    # Handle midyear classes (add "TWThF" for days)
    df.loc[df['Semester'] == "Midyear", 'Day'] = 'TWThF'
    
    # Drop rows with missing critical values
    df = df.dropna(subset=["Prof", "Day", "Start_time", "End_time"]).copy()
    
    # Convert times to minutes since midnight
    df["Start_minutes"] = df["Start_time"].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    df["End_minutes"] = df["End_time"].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    
    return df

@st.cache_resource
def load_model():
    # Load the pre-trained model
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_label_encoders(_df):
    # Create and fit label encoders based on actual data
    le_semester = LabelEncoder()
    le_semester.fit(_df['Semester'].unique())
    
    le_prof = LabelEncoder()
    le_prof.fit(_df['Prof'].unique())
    
    return {'Semester': le_semester, 'Prof': le_prof}

# Load resources
try:
    model_df = load_data()
    best_rf_model = load_model()
    label_encoders = load_label_encoders(model_df)
    
    # Get unique values for filters
    unique_profs = sorted(model_df['Prof'].unique())
    unique_numbers = sorted(model_df['Number'].unique())
    unique_days = sorted(model_df['Day'].unique())
    unique_rooms = sorted(model_df['Room'].unique())
    unique_semesters = sorted(model_df['Semester'].unique())
    unique_years = sorted(model_df['Year'].unique())
    unique_start_times = sorted(model_df['Start_time'].unique())
    unique_end_times = sorted(model_df['End_time'].unique())
    
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# Prediction function
def predictor(class_number, day, room, semester, start_time, end_time):
    try:
        # Convert semester using LabelEncoder
        semester_encoded = label_encoders['Semester'].transform([semester])[0]
        
        # Convert times to minutes since midnight
        start_minutes = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
        end_minutes = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
        
        # Build base input dict
        input_data = {
            "Semester": semester_encoded,
            "Start_minutes": start_minutes,
            "End_minutes": end_minutes,
            "Year": datetime.now().year  # Use current year as default
        }
        
        # Handle one-hot encoded categorical variables
        # Create all possible columns with 0 values
        for day_val in unique_days:
            input_data[f"Day_{day_val}"] = 1 if day_val == day else 0
            
        for num_val in unique_numbers:
            input_data[f"Number_{num_val}"] = 1 if str(num_val) == str(class_number) else 0
            
        for room_val in unique_rooms:
            input_data[f"Room_{room_val}"] = 1 if room_val == room else 0
        
        # Create dataframe
        input_df = pd.DataFrame([input_data])
        
        # Ensure we have all columns the model expects
        if hasattr(best_rf_model, 'feature_names_in_'):
            model_columns = best_rf_model.feature_names_in_
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_columns]
        
        # Prediction
        pred_num = best_rf_model.predict(input_df)[0]
        return label_encoders['Prof'].inverse_transform([pred_num])[0]
    
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Create tabs
tab1, tab2 = st.tabs(["Professor Predictor", "Schedule Dashboard"])

with tab1:
    st.header("Professor Predictor")
    st.write("Predictor is made using Random Forest Classifier (33.26% accuracy). Enter class details to predict the most likely professor.")
    
    # Create input form
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            class_number = st.selectbox("Class Number", unique_numbers)
            day = st.selectbox("Day", unique_days)
            room = st.selectbox("Room", unique_rooms)
            
        with col2:
            semester = st.selectbox("Semester", unique_semesters)
            start_time = st.selectbox("Start Time", unique_start_times)
            end_time = st.selectbox("End Time", unique_end_times)
            
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            st.write("")  # Spacer
            st.write("")  # Spacer
            st.write("")  # Spacer
            st.write("")  # Spacer
            st.write("")  # Spacer
            submit_button = st.form_submit_button("Predict Professor")
    
    if submit_button:
        # Validate time format
        time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
        if not time_pattern.match(start_time) or not time_pattern.match(end_time):
            st.error("Please enter times in HH:MM format (e.g., 14:30)")
        else:
            try:
                predicted_prof = predictor(class_number, day, room, semester, start_time, end_time)
                st.success(f"Predicted Professor: {predicted_prof}")
                
                # Show similar classes from actual data
                similar_classes = model_df[
                    (model_df['Number'] == class_number) & 
                    (model_df['Day'] == day) & 
                    (model_df['Room'] == room)
                ]
                
                if not similar_classes.empty:
                    st.write("Historical classes with these details:")
                    st.dataframe(similar_classes.sort_values('Year', ascending=False))
                else:
                    st.write("No historical classes found with these exact details.")
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

with tab2:
    st.header("Schedule Dashboard")
    st.write("Filter classes to visualize professor schedules")
    
    # Create filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prof_filter = st.selectbox("Professor", ["All"] + unique_profs)
            class_filter = st.selectbox("Class Number", ["All"] + unique_numbers)
            
        with col2:
            day_filter = st.multiselect("Day(s)", unique_days)
            semester_filter = st.multiselect("Semester", unique_semesters)
            
        with col3:
            room_filter = st.multiselect("Room(s)", unique_rooms)
            year_filter = st.multiselect("Year", unique_years)
    
    # Apply filters
    filtered_df = model_df[['Number', 'Day', 'Room', 'Prof', 'Year', 'Semester', 'Start_time', 'End_time']]
    
    if prof_filter != "All":
        filtered_df = filtered_df[filtered_df['Prof'] == prof_filter]
    if class_filter != "All":
        filtered_df = filtered_df[filtered_df['Number'] == class_filter]
    if day_filter:
        filtered_df = filtered_df[filtered_df['Day'].isin(day_filter)]
    if semester_filter:
        filtered_df = filtered_df[filtered_df['Semester'].isin(semester_filter)]
    if room_filter:
        filtered_df = filtered_df[filtered_df['Room'].isin(room_filter)]
    if year_filter:
        filtered_df = filtered_df[filtered_df['Year'].isin(year_filter)]
    
    # Show filtered data
    st.write(f"Found {len(filtered_df)} classes matching your criteria")
    
    if not filtered_df.empty:
        # Display data table
        st.dataframe(filtered_df.sort_values(['Year', 'Semester']))
        
        # Create visualizations
        st.subheader("Class Schedule Visualization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Start time distribution
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            filtered_df['Start_time'].value_counts().sort_index().plot(kind='bar', ax=ax1)
            ax1.set_title("Class Start Time Distribution")
            ax1.set_xlabel("Start Time")
            ax1.set_ylabel("Number of Classes")
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
            # Semester distribution
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            filtered_df['Semester'].value_counts().plot(kind='bar', ax=ax3)
            ax3.set_title("Semester Distribution")
            ax3.set_xlabel("Semester")
            ax3.set_ylabel("Number of Classes")
            st.pyplot(fig3)
            
        with col2:
            # Day of week distribution
            fig2, ax2 = plt.subplots(figsize=(3, 1.5))
            filtered_df['Day'].value_counts().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 5}, ax=ax2)
            ax2.set_title("Day of Week Distribution")
            ax2.set_ylabel("")
            st.pyplot(fig2)
            
            # Year distribution
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            filtered_df['Year'].value_counts().sort_index().plot(kind='line', marker='o', ax=ax4)
            ax4.set_title("Classes Per Year")
            ax4.set_xlabel("Year")
            ax4.set_ylabel("Number of Classes")
            st.pyplot(fig4)

        with col3:
            # Room distribution
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            filtered_df['Room'].value_counts().sort_index().plot(kind='bar', ax=ax5)
            ax5.set_title("Room Distribution")
            ax5.set_ylabel("Number of Rooms")
            st.pyplot(fig5)
    else:

        st.warning("No classes match your filters. Try adjusting your criteria.")
