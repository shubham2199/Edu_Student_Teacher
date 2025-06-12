# 🎓 Edu-Mentor AI

**Edu-Mentor AI** is an intelligent student performance analysis and risk prediction system powered by Machine Learning and Generative AI. It provides separate dashboards for students and teachers with personalized insights, risk prediction, and LLM-based recommendations and chat.

---

## 🚀 Features

- 🔐 **Secure Login** for Students and Teachers
- 📊 **ML-Powered Risk Score** Prediction
- 🧠 **LLM Agent** for answering student-specific questions
- 📚 **Subject-Wise Teacher Dashboards**
- ✍️ **Teacher Feedback System** (with update and history support)
- 💬 **Student Dashboard with Feedback Display**
- 📤 **Export Reports** (for teachers/admin)
- 🛠️ Admin Dashboard (optional)

---

## 🧠 Technologies Used

- **Python 3.10+**
- **Streamlit** (UI)
- **Scikit-learn** (ML model)
- **XGBoost**
- **LangChain / GROQ API** (LLM)
- **Pandas, NumPy**
- **Joblib** (Model Serialization)
- **dotenv** (Environment management)

---

## 📂 Folder Structure

Edu-Mentor-AI/
│
├── data/
│ ├── edu_mentor_dataset_final.csv      # Student Data.
│ └── student_feedback.csv              # Feedback History.
│
├── model/
│ └── risk_score_model.pkl              # Train Model and create PKL file. 
│
├── train/
| └── train_and_save_model.py           # Train models and save into a PKL file.
|
├── .env                                # Store GROQ API key and path the PKL, CSV file.
├── app.py                              # Main File to run this file with "streamlit run app.py".
├── student_dashboard.py                # Student Dashboard.
├── teacher_dashboard.py                # Teachers Dashboard.
├── llm_utils.py                        # Prompt Genaration.
└── README.md