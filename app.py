import streamlit as st

from src.screens.home_screen import home_screen
from src.screens.teacher_screen import teacher_screen
from src.screens.student_screen import student_screen
from src.components.dialog_auto_enroll import auto_enroll_dialog


def main():
    st.set_page_config(
        page_title="SnapClass - Making Attendance faster using AI",
        page_icon="https://i.ibb.co/YTYGn5qV/logo.png"
    )

    # session states
    if "login_type" not in st.session_state:
        st.session_state.login_type = None

    if "show_auto_enroll" not in st.session_state:
        st.session_state.show_auto_enroll = False

    # GET JOIN CODE
    join_code = st.query_params.get("join_code")

    # FORCE STUDENT SCREEN
    if join_code and st.session_state.login_type != "student":
        st.session_state.login_type = "student"

    # SCREEN ROUTING
    match st.session_state.login_type:
        case "teacher":
            teacher_screen()

        case "student":
            student_screen()

        case None:
            home_screen()

    # AUTO ENROLL AFTER LOGIN
    if (
        join_code
        and st.session_state.get("is_logged_in")
        and st.session_state.get("user_role") == "student"
    ):
        auto_enroll_dialog(join_code)


main()
