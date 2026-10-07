import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

import config
import database as db
from face_engine import FaceEngine, Gallery


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="AI Face Attendance",
    page_icon="📸",
    layout="wide"
)

st.title("📸 AI Face Attendance System")
st.caption("AI-powered face recognition and attendance management")


# --------------------------------------------------
# INITIALIZE DATABASE + AI
# --------------------------------------------------

@st.cache_resource
def load_engine():
    return FaceEngine()


@st.cache_resource
def load_gallery():
    users, rows = db.load_gallery()

    gallery = Gallery()
    gallery.load(users, rows)

    return gallery


db.init_db()

try:
    engine = load_engine()
    gallery = load_gallery()
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def read_camera_image(uploaded_file):
    """Convert Streamlit camera image into OpenCV BGR image."""
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_faces(frame):
    faces = engine.detect(frame)
    return faces


def draw_faces(frame, faces, labels=None):
    """Draw face boxes and labels."""
    output = frame.copy()

    for i, face in enumerate(faces):
        x, y, w, h = [int(v) for v in face[:4]]

        color = (0, 200, 0)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        if labels and i < len(labels):
            label = labels[i]

            cv2.rectangle(
                output,
                (x, max(0, y - 35)),
                (x + max(150, len(label) * 10), y),
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    return output


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📋 Menu")

page = st.sidebar.radio(
    "Choose an option",
    [
        "🏠 Dashboard",
        "➕ Register Person",
        "📸 Mark Attendance",
        "📊 Attendance Records",
        "👥 Manage People"
    ]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    users = db.get_users()
    attendance = db.get_attendance(
        datetime.now().strftime("%Y-%m-%d")
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Registered People",
            len(users)
        )

    with col2:
        st.metric(
            "Today's Attendance",
            len(attendance)
        )

    with col3:
        st.metric(
            "System Status",
            "🟢 Online"
        )

    st.divider()

    st.subheader("Today's Attendance")

    if attendance:
        st.dataframe(
            attendance,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No attendance recorded today.")

    st.markdown("""
    ### How it works

    **1. Register a person**  
    Capture their face using the camera.

    **2. Mark attendance**  
    Take a camera snapshot. The AI detects and recognizes the face.

    **3. Attendance is stored**  
    The system records the person's name, date, time and status.

    **4. Export records**  
    Attendance records can be downloaded as CSV.
    """)


# ==================================================
# REGISTER PERSON
# ==================================================

elif page == "➕ Register Person":

    st.header("➕ Register New Person")

    st.write(
        "Enter the person's details and capture one clear face image."
    )

    name = st.text_input(
        "Full Name"
    )

    emp_id = st.text_input(
        "ID / Roll Number"
    )

    department = st.text_input(
        "Department / Class"
    )

    camera = st.camera_input(
        "📷 Take a photo for registration"
    )

    if camera:

        frame = read_camera_image(camera)

        faces = detect_faces(frame)

        if len(faces) == 0:

            st.error(
                "❌ No face detected. Please take another photo."
            )

        elif len(faces) > 1:

            st.error(
                "❌ Multiple faces detected. Please make sure only one person is visible."
            )

        else:

            face = faces[0]

            x, y, w, h = [int(v) for v in face[:4]]

            if min(w, h) < config.MIN_FACE_SIZE:

                st.warning(
                    "⚠️ Face is too small. Move closer to the camera."
                )

            elif face[14] < 0.85:

                st.warning(
                    "⚠️ Please face the camera more directly."
                )

            else:

                preview = draw_faces(
                    frame,
                    faces,
                    ["Face detected"]
                )

                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Face detected successfully",
                    use_container_width=True
                )

                if st.button(
                    "✅ Register Person",
                    type="primary"
                ):

                    if not name.strip() or not emp_id.strip():

                        st.error(
                            "Name and ID are required."
                        )

                    elif db.emp_id_exists(emp_id.strip()):

                        st.error(
                            "This ID is already registered."
                        )

                    else:

                        embedding = engine.embed(
                            frame,
                            face
                        )

                        try:

                            db.add_user(
                                name.strip(),
                                emp_id.strip(),
                                department.strip(),
                                [embedding]
                            )

                            st.success(
                                f"🎉 {name} registered successfully!"
                            )

                            # Refresh cached gallery
                            load_gallery.clear()

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Registration failed: {e}"
                            )


# ==================================================
# MARK ATTENDANCE
# ==================================================

elif page == "📸 Mark Attendance":

    st.header("📸 Mark Attendance")

    st.write(
        "Take a clear camera snapshot. The AI will detect and identify the person."
    )

    if len(gallery) == 0:

        st.warning(
            "⚠️ No people are registered yet. Register someone first."
        )

    else:

        camera = st.camera_input(
            "📷 Look at the camera and take a photo"
        )

        if camera:

            frame = read_camera_image(camera)

            faces = detect_faces(frame)

            if len(faces) == 0:

                st.error(
                    "❌ No face detected."
                )

            else:

                labels = []

                recognized = []

                for face in faces:

                    if min(
                        int(face[2]),
                        int(face[3])
                    ) < config.MIN_FACE_SIZE:

                        labels.append("Move closer")
                        continue

                    embedding = engine.embed(
                        frame,
                        face
                    )

                    uid, score = gallery.identify(
                        embedding
                    )

                    if uid is None:

                        labels.append(
                            f"Unknown ({score:.2f})"
                        )

                    else:

                        person = gallery.users[uid]

                        name = person["name"]

                        labels.append(
                            f"{name} ({score:.2f})"
                        )

                        recognized.append(
                            (uid, name, score)
                        )

                output = draw_faces(
                    frame,
                    faces,
                    labels
                )

                st.image(
                    cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                    caption="Recognition result",
                    use_container_width=True
                )

                if not recognized:

                    st.error(
                        "❌ No registered person recognized."
                    )

                else:

                    for uid, name, score in recognized:

                        st.success(
                            f"👤 Recognized: **{name}**  |  "
                            f"Similarity: **{score:.2f}**"
                        )

                        today = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                        marked = db.marked_user_ids(
                            today
                        )

                        if uid in marked:

                            st.info(
                                f"ℹ️ {name}'s attendance is already marked today."
                            )

                        else:

                            record = db.mark_attendance(uid)

                            if record:

                                if record["status"] == "Late":

                                    st.warning(
                                        f"⏰ {name} marked **Late** "
                                        f"at {record['time']}"
                                    )

                                else:

                                    st.success(
                                        f"✅ {name} marked **Present** "
                                        f"at {record['time']}"
                                    )

                                load_gallery.clear()


# ==================================================
# ATTENDANCE RECORDS
# ==================================================

elif page == "📊 Attendance Records":

    st.header("📊 Attendance Records")

    date = st.date_input(
        "Select date",
        datetime.now()
    )

    day = date.strftime(
        "%Y-%m-%d"
    )

    records = db.get_attendance(day)

    if records:

        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True
        )

        csv_data = "\n".join(
            [
                ",".join(
                    str(row[col])
                    for col in row
                )
                for row in records
            ]
        )

        st.download_button(
            "⬇️ Download Attendance CSV",
            data=csv_data,
            file_name=f"attendance_{day}.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No attendance records found for this date."
        )


# ==================================================
# MANAGE PEOPLE
# ==================================================

elif page == "👥 Manage People":

    st.header("👥 Registered People")

    users = db.get_users()

    if not users:

        st.info(
            "No people are registered."
        )

    else:

        for user in users:

            with st.container(border=True):

                col1, col2, col3 = st.columns(
                    [3, 2, 1]
                )

                with col1:

                    st.write(
                        f"### 👤 {user['name']}"
                    )

                    st.write(
                        f"ID: {user['emp_id']}"
                    )

                with col2:

                    st.write(
                        f"Department: {user['department']}"
                    )

                    st.write(
                        f"Face samples: {user['samples']}"
                    )

                with col3:

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{user['id']}"
                    ):

                        db.delete_user(
                            user["id"]
                        )

                        load_gallery.clear()

                        st.success(
                            "Person deleted."
                        )

                        st.rerun()