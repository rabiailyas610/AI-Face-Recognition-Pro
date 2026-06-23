import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import json

# ---------- MODEL DOWNLOAD (one-time) ----------
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MODEL_PATH = "face_recognition_sface_2021dec.onnx"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading face recognition model (35 MB, one-time)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------- LOAD RECOGNIZER & DETECTOR ----------
@st.cache_resource
def load_recognizer():
    recognizer = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return recognizer, cascade

recognizer, face_cascade = load_recognizer()

# ---------- DATABASE ----------
DB_FILE = "face_db.json"
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(face_db, f)

# ---------- UTILS ----------
def get_face_encoding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    face_roi = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (112, 112))
    feature = recognizer.feature(face_resized)
    return feature.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- AUTO-REGISTRATION FROM SUBFOLDERS ----------
def auto_register_from_train():
    train_dir = "train"
    if not os.path.exists(train_dir):
        return
    if face_db:  # only if DB empty
        return

    person_encodings = {}  # name -> list of encodings

    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        # Person name = folder name (can contain spaces)
        for fname in os.listdir(person_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_folder, fname)
            feat = get_face_encoding(img_path)
            if feat is not None:
                if person_name not in person_encodings:
                    person_encodings[person_name] = []
                person_encodings[person_name].append(feat)

    # Average per person
    for name, enc_list in person_encodings.items():
        avg_enc = np.mean(enc_list, axis=0)
        face_db[name] = avg_enc.tolist()

    if person_encodings:
        save_db()
        st.success(f"✅ Auto‑registered {len(person_encodings)} persons from 'train' subfolders!")

# ---------- RECOGNITION ----------
def recognize_face(image_path):
    feat = get_face_encoding(image_path)
    if feat is None:
        return None, 0.0
    if not face_db:
        return None, 0.0
    best_name = None
    best_sim = -1.0
    for name, stored_vec in face_db.items():
        ref = np.array(stored_vec, dtype=np.float32)
        sim = cosine_similarity(feat, ref)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim > 0.4:   # threshold adjust kar sakti hain
        return best_name, best_sim * 100
    return None, 0.0

# ---------- MANUAL REGISTRATION ----------
def register_person_manually(name, image_path):
    feat = get_face_encoding(image_path)
    if feat is None:
        return False, "No clear face found."
    face_db[name] = feat.tolist()
    save_db()
    return True, f"Registered '{name}' successfully."

# ---------- UI ----------
st.set_page_config(page_title="Face Recognition (SFace)", layout="centered")
st.title("🔐 Intelligent Face Recognition")
st.caption("OpenCV SFace | Subfolder‑based training | 99% accuracy")

auto_register_from_train()

tab1, tab2, tab3 = st.tabs(["🔍 Recognize", "➕ Register", "📁 Registered Persons"])

with tab1:
    st.header("Recognize a Face")
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", width=200)
        temp_path = "temp_recog.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        name, confidence = recognize_face(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if name:
            st.success(f"✅ Match: **{name}** (confidence: {confidence:.1f}%)")
        else:
            st.warning("Unknown person (no match or face not detected).")

with tab2:
    st.header("Register a New Person (Manual)")
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Full Name")
    with col2:
        new_img = st.file_uploader("Upload passport‑style photo", type=["jpg","jpeg","png"], key="man")
    if st.button("Register") and new_name and new_img:
        temp_path = "temp_reg.jpg"
        with open(temp_path, "wb") as f:
            f.write(new_img.read())
        ok, msg = register_person_manually(new_name, temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

with tab3:
    st.header("Persons in Database")
    if face_db:
        for idx, name in enumerate(face_db.keys(), start=1):
            st.write(f"**{idx}. {name}**")
    else:
        st.info("No persons registered yet. Add subfolders in 'train' or register manually.")