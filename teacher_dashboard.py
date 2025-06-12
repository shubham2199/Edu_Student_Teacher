import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from llm_utils import ask_student_agent
from dotenv import load_dotenv

load_dotenv()


feedback_path = os.getenv("FEEDBACK_FILE")

@st.cache_resource
def load_data():
    dataset_path  = os.getenv("DATASET_FILE")
    model_path  = os.getenv("PKL_FILE") 

    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    df = pd.read_csv(dataset_path)
    artifacts = joblib.load(model_path)
    return df, artifacts


def teacher_main():
    df, art = load_data()
    model = art["model"]
    scaler = art["scaler"]
    feat_cols = art["feature_columns"]
    label_encoders = art["label_encoders"]

    # Ensure session keys exist
    if "role" not in st.session_state:
        st.session_state.role = None
    if "teacher_subject" not in st.session_state:
        st.session_state.teacher_subject = None

    # TEACHER LOGIN PAGE
    if not st.session_state.get("authenticated", False):
        st.title("👩‍🏫 Teacher Login")
        subjects = {
            "math": "math_pass",
            "science": "science_pass",
            "english": "english_pass",
            "history": "history_pass"
        }
        sub = st.selectbox("Select Subject wise Teacher", list(subjects.keys()))
        pw = st.text_input("Password", type="password")

        col1, col2, col3 = st.columns([6, 2, 1])
        with col1:
            if st.button("🔓 Login"):
                if pw == subjects[sub]:
                    st.session_state.teacher_subject = sub
                    st.session_state.authenticated = True  # ✅ instead of changing role
                    st.rerun()
                else:
                    st.error("❌ Invalid teacher password")             
        with col2:
            if st.button("🔙 Back to Home"):
                st.session_state.role = None
                st.session_state.authenticated = False
                st.session_state.teacher_subject = None
                st.rerun()
                    
        return  # <- Prevents blank page on initial load after rerun

    # TEACHER DASHBOARD
    st.title(f"📘 {st.session_state.teacher_subject.capitalize()} Teacher Dashboard")
    
    sub_col = st.session_state.teacher_subject + "_grade"
    low_students = df[df[sub_col] < 60]

    st.markdown(f"### 📋 Students with less marks in **{st.session_state.teacher_subject.capitalize()}**")
    st.dataframe(low_students[["student_id", "student_name", sub_col]])

    sid = st.text_input("🔎 Enter Student ID to check risk")
    if sid:
        student_row = df[df["student_id"].astype(str) == sid]
        if not student_row.empty:
            student_row = student_row.iloc[0]
            sub_score = student_row[sub_col]
            st.write(f"🧑 Student Name: **{student_row['student_name']}**")
            st.write(f"📊 {st.session_state.teacher_subject.capitalize()} Marks: **{sub_score}**")

            try:
                input_df = pd.DataFrame([student_row[feat_cols]])

                # Label encode
                for col in label_encoders:
                    val = input_df[col].values[0]
                    input_df[col] = label_encoders[col].transform([val])

                X_scaled = scaler.transform(input_df)
                score = float(np.clip(model.predict(X_scaled)[0], 0, 100))
                risk = "⚠️ At Risk" if score > 60 else "✅ Not At Risk"

                st.metric("📈 Risk Score", f"{score:.2f}")
                st.metric("🔍 Risk Status", risk)
                
                # 👇 After displaying the risk score
                st.markdown("### 🤖 Ask a question about this student")
                user_question = st.text_input("Enter your question")

                if user_question:
                    student_data_dict = student_row[feat_cols].to_dict()
                    student_data_dict.update({
                        "student_id": student_row["student_id"],
                        "student_name": student_row["student_name"],
                        "risk_score": f"{score:.2f}"
                    })

                    # Convert dict to readable format
                    student_context = ", ".join(f"{k}: {v}" for k, v in student_data_dict.items())

                    try:
                        response = ask_student_agent(student_context, user_question)
                        st.markdown(f"🕵🏻 **Agent:** {response}")
                    except Exception as e:
                        st.error(f"⚠️ Agent failed to respond: {e}")
                
                        
                FEEDBACK_FILE = feedback_path

                # Create feedback CSV if not exists
                if not os.path.exists(FEEDBACK_FILE):
                    pd.DataFrame(columns=[
                        "student_id", "student_name", "teacher_subject", 
                        "feedback", "timestamp"
                    ]).to_csv(FEEDBACK_FILE, index=False)

                # Feedback entry section
                st.markdown("### 📝 Submit Feedback for this Student")
                feedback_text = st.text_area("Write your feedback here:")

                if st.button("📨 Submit Feedback"):
                    if feedback_text.strip():
                        feedback_entry = {
                            "student_id": student_row["student_id"],
                            "student_name": student_row["student_name"],
                            "teacher_subject": st.session_state.teacher_subject,
                            "feedback": feedback_text.strip(),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Append to CSV
                        feedback_df = pd.read_csv(FEEDBACK_FILE)
                        feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback_entry])], ignore_index=True)
                        feedback_df.to_csv(FEEDBACK_FILE, index=False)

                        st.success("✅ Feedback submitted successfully.")
                    else:
                        st.warning("⚠️ Please enter some feedback before submitting.")
                        
                        
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
        else:
            st.error("❌ Student ID not found")
            
        col1, col2, col3 = st.columns([6, 2, 2])
        with col3:
            if st.button("🔒 Logout"):
                st.session_state.role = None
                st.session_state.authenticated = False
                st.session_state.teacher_subject = None
                st.rerun()
