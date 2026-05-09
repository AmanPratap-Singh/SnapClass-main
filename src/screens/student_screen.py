import streamlit as st
import time
import numpy as np
from PIL import Image

from src.Ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

from src.pipelines.face_pipeline import (
    predict_attendance,
    get_face_embeddings,
    train_classifier
)

from src.pipelines.voice_pipeline import get_voice_embeddings
from src.database.db import get_all_students, create_student, get_student_subjects, get_student_attendance, unenroll_student_tosubject


# ---------------------------
# Dashboard after login
# ---------------------------
def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']

    c1, c2 = st.columns(2, gap="xxlarge", vertical_alignment="center")

    with c1:
        header_dashboard()

    with c2:
        st.subheader(f"""Welcome, {student_data['name']} """)
        if st.button(
            "Logout",
            type="secondary",
            key="teacher_login_back_btn",
            shortcut="control+backspace"
        ):
            st.session_state["is_logged_in"] = False
            del st.session_state.student_data
            st.rerun()

    st.space()

    c1, c2 = st.columns(2)
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()

    st.divider()

    with st.spinner('Loading your enrolled subjects..'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}

    for log in logs:
        sid = log['subject_id']

        if sid not in stats_map:
            stats_map[sid] = {"total":0, "attended":0}

        stats_map[sid]['total'] += 1

        if log.get('is_present'):
            stats_map[sid]['attended'] += 1 

    cols = st.columns(2)
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']

        stats = stats_map.get(sid, {"total":0, "attended":0})
        def unenroll_button():
                if st.button('Unenroll from this course', type='tertiary', width='stretch', key=f"unenroll_{sid}"):
                    unenroll_student_tosubject(student_id, sid)
                    st.toast(f"Unerolled from {sub['name']} Sucessfully!")   
                    st.rerun()    

        with cols[i % 2]:
            subject_card(
                name = sub['name'],
                code = sub['subject_code'],
                section =   sub['section'],
                stats = [
                    ('📆', 'Total', stats['total']),
                    ('✅', 'Attended', stats['attended'])
                ],
                footer_callback=unenroll_button
            )


    footer_dashboard()


# ---------------------------
# Main Student Screen
# ---------------------------
def student_screen():

    style_background_dashboard()
    style_base_layout()

    # ---------------------------
    # Session defaults
    # ---------------------------
    if "show_registration" not in st.session_state:
        st.session_state.show_registration = False

    if "student_data" in st.session_state:
        student_dashboard()
        return

    # ---------------------------
    # Header
    # ---------------------------
    c1, c2 = st.columns(2, gap="large")

    with c1:
        header_dashboard()

    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="student_back_btn"
        ):
            st.session_state["login_type"] = None
            st.rerun()

    st.header("Login Using FaceID")

    st.write("")

    # ---------------------------
    # Camera Input
    # ---------------------------
    photo_source = st.camera_input("Position your face in the center")

    # ---------------------------
    # Login Flow
    # ---------------------------
    if photo_source is not None:

        try:
            img = Image.open(photo_source).convert("RGB")
            img = img.resize((640, 480))
            img = np.array(img)

            with st.spinner("Scanning details..."):

                detected, all_ids, num_faces = predict_attendance(img)

            # Debug info
            # st.write("Faces found:", num_faces)
            # st.write("Known IDs:", all_ids)
            # st.write("Detected:", detected)

            # ---------------------------
            # Face checks
            # ---------------------------
            if num_faces == 0:
                st.warning("No face found. Please retake photo.")

            elif num_faces > 1:
                st.markdown(
                """
                <style>
                div[data-testid="stAlert"] p {
                    color: black;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
                st.warning("Multiple faces found. Only one person should appear.")

            else:
                # Exactly one face
                if detected:

                    student_id = list(detected.keys())[0]

                    all_students = get_all_students()

                    # Convert both sides to string for safe matching
                    student = next(
                        (
                            s for s in all_students
                            if str(s["student_id"]) == str(student_id)
                        ),
                        None
                    )

                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.student_data = student

                        st.success(f"Welcome Back {student['name']}")

                        time.sleep(1)
                        st.rerun()

                    else:
                        st.info("Face recognized but student not found.")
                        st.session_state.show_registration = True

                else:
                    st.info("Face not recognized. Register below.")
                    st.session_state.show_registration = True

        except Exception as e:
            st.error(f"Login error: {str(e)}")

    # ---------------------------
    # Registration Section
    # ---------------------------
    if st.session_state.show_registration:

        with st.container(border=True):

            st.header("Register New Profile")

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Aman"
            )

            st.subheader("Optional Voice Enrollment")
            st.info("Record a short phrase for future attendance")

            audio_data = None

            try:
                audio_data = st.audio_input(
                    "Say: I am present, my name is Aman"
                )
            except Exception:
                pass

            if st.button("Create Account", type="primary"):

                if not new_name.strip():
                    st.warning("Please enter your name")
                    st.stop()

                if photo_source is None:
                    st.warning("Please capture photo first")
                    st.stop()

                try:
                    with st.spinner("Creating profile..."):

                        img = Image.open(photo_source).convert("RGB")
                        img = img.resize((640, 480))
                        img = np.array(img)

                        encodings = get_face_embeddings(img)

                        if len(encodings) == 0:
                            st.error("No face detected for registration")
                            st.stop()

                        face_emd = encodings[0].tolist()

                        voice_emd = None
                        if audio_data is not None:
                            try:
                                voice_emd = get_voice_embeddings(
                                    audio_data.read()
                                )
                            except Exception:
                                pass

                        response_data = create_student(
                            new_name,
                            face_embedding=face_emd,
                            voice_embedding=voice_emd
                        )

                        if response_data:

                            train_classifier()

                            st.session_state.is_logged_in = True
                            st.session_state.user_role = "student"
                            st.session_state.student_data = response_data[0]
                            st.session_state.show_registration = False

                            st.success(f"Profile Created! Hi {new_name}")

                            time.sleep(1)
                            st.rerun()

                        else:
                            st.error("Failed to create account")

                except Exception as e:
                    st.error(f"Registration error: {str(e)}")

    # ---------------------------
    # Footer
    # ---------------------------
    footer_dashboard()
