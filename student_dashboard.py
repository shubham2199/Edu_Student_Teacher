import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from llm_utils import ask_student_agent
from dotenv import load_dotenv

load_dotenv()

feedback_path = os.getenv("FEEDBACK_FILE")


@st.cache_resource
def load_data():
    
    dataset_path = os.getenv("DATASET_FILE")
    model_path = os.getenv("PKL_FILE")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    df = pd.read_csv(dataset_path)
    artifacts = joblib.load(model_path)
    return df, artifacts

def student_main():
    df, art = load_data()
    model, scaler = art["model"], art["scaler"]
    encoders, feat_cols = art["label_encoders"], art["feature_columns"]
    metrics = art["metrics"]

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'student_id' not in st.session_state:
        st.session_state.student_id = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    
    if not st.session_state.logged_in:
        st.title("🧑‍🎓 Student Login")
        sid = st.text_input("Student ID")
        pwd = st.text_input("Password", type="password")
        
        col1, col2, col3 = st.columns([6, 2, 1])

        with col1:
            if st.button("🔓 Login"):
                row = df[df["student_id"].astype(str) == sid]
                if not row.empty and row.iloc[0]["password"] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.student_id = sid
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with col2:
            if st.button("🔙 Back to Home"):
                st.session_state.role = None
                st.session_state.logged_in = False
                st.rerun() 
    else:
        row = df[df["student_id"].astype(str) == st.session_state.student_id].iloc[0]
        st.title(f"🎓 {row.student_id} : {row.student_name}'s Dashboard")
                
        rdata = row.copy()
        for col, le in encoders.items():
            rdata[col] = le.transform([rdata[col]])

        X = scaler.transform(pd.DataFrame([rdata[feat_cols]]))
        score = float(np.clip(model.predict(X)[0], 0, 100))
        risk = "⚠️ At Risk" if score > 60 else "✅ Not At Risk"

        st.metric("📈 Risk Score", f"{score:.2f}")
        st.metric("🔍 Risk Status", risk)

        st.markdown("---")
        st.markdown("### 🕵🏻 LLM Recommendations")
        important_data = {
            "Math": row["math_grade"],
            "Science": row["science_grade"],
            "English": row["english_grade"],
            "History": row["history_grade"],
            "Assignment": row["assignment_completion"],
            "Engagement Score": row["engagement_score"],
            "Attendance (%)": row["attendance_ratio"],
            "Lms Test Scores": row["lms_test_scores"]
        }

        recommendation_prompt = (
        "You are an academic advisor AI. Analyze the following student data and provide detailed improvement suggestions "
        "for EACH subject or area where the performance is below 70%. If all are good, just say so. "
        "Include practical steps like study techniques, time management tips, or recommended resources.\n\n"
        "...Format your response in bullet points for clarity."
        f"Student Performance Data:\n{important_data}"
        )

        reco = ask_student_agent(context_data=important_data, user_prompt=recommendation_prompt)
        st.markdown(reco)
        
        FEEDBACK_FILE = feedback_path

        if os.path.exists(FEEDBACK_FILE):
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            student_feedbacks = feedback_df[feedback_df["student_id"] == row["student_id"]]

            if not student_feedbacks.empty:
                st.markdown("### 💬 Teacher Feedback")
                for _, fb in student_feedbacks.iterrows():
                    st.info(f"""🧑‍🏫 **{fb['teacher_subject'].capitalize()} Teacher**  
        📅 {fb['timestamp']}  
        📝 {fb['feedback']}""")
            else:
                st.markdown("### 💬 Teacher Feedback")
                st.write("No feedback submitted yet.")
          
                
        col1, col2, col3 = st.columns([6, 2, 2])
        # Logout button top-right
        with col3:
            if st.button("🔒 Logout"):
                st.session_state.role = None
                st.rerun()


        st.markdown("---")
        st.header("🤖 Ask about this student")
        user_q = st.chat_input("Ask something...")
        if user_q:
            ctx = {k: row[k] for k in row.index if pd.notnull(row[k])}
            ans = ask_student_agent(ctx, user_q)
            st.session_state.chat_history.append(("user", user_q))
            st.session_state.chat_history.append(("assistant", ans))

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)