import streamlit as st
from student_dashboard import student_main
from teacher_dashboard import teacher_main

st.set_page_config(page_title="Edu-Mentor Home", layout="centered")

# Reset app to main screen if logging out
if st.session_state.get("role") in [None, ""]:
    st.session_state.role = None
    st.session_state.teacher_subject = None
    st.session_state.student_id = None

# MAIN PAGE
if st.session_state.get("role") is None:
    st.title("🎓 Edu-Mentor AI")
    st.write("Select who you are to continue:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧑‍🎓 Student"):
            st.session_state.role = "student"
            st.rerun()
    with col2:
        if st.button("👩‍🏫 Teacher"):
            st.session_state.role = "teacher"
            st.rerun()

# REDIRECT TO STUDENT OR TEACHER DASHBOARD
elif st.session_state.role == "student":
    student_main()
elif st.session_state.role == "teacher":
    teacher_main()
