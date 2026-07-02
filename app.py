import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import json
import ssl
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Face Recognition Pro",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SESSION STATE
# ============================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_recognitions" not in st.session_state:
    st.session_state.total_recognitions = 0

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history():
    history_to_save = []
    for entry in st.session_state.history:
        entry_copy = entry.copy()
        if 'accuracy' in entry_copy:
            entry_copy['accuracy'] = float(entry_copy['accuracy'])
        history_to_save.append(entry_copy)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_to_save, f, indent=2)

if not st.session_state.history:
    st.session_state.history = load_history()
    st.session_state.total_recognitions = len(st.session_state.history)

# ============================================
# CSS (Default Streamlit Layout)
# ============================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background: #0b0f1a; }
    header h1 { display: none !important; }
    
    section[data-testid="stSidebar"] {
        background: rgba(11,15,26,0.98) !important;
        border-right: 1px solid rgba(255,255,255,0.04);
        padding: 16px 12px !important;
    }
    
    .sidebar-stat {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 6px;
        padding: 5px 10px;
        margin: 2px 0;
        border-left: 3px solid #4f46e5;
    }
    .sidebar-stat .label { color: #64748b; font-size: 9px; text-transform: uppercase; }
    .sidebar-stat .value { color: #f1f5f9; font-size: 14px; font-weight: 600; }
    
    .history-item {
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #4f46e5;
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 11px;
    }
    .history-time { color: #64748b; font-size: 9px; }
    .history-name { color: #f1f5f9; font-weight: 600; }
    
    .dash-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        transition: 0.2s;
    }
    .dash-card:hover {
        background: rgba(255,255,255,0.04);
        transform: translateY(-2px);
    }
    .dash-card .icon { font-size: 28px; }
    .dash-card .value { font-size: 34px; font-weight: 700; color: #f1f5f9; }
    .dash-card .label { color: #64748b; font-size: 11px; text-transform: uppercase; }
    .dash-card .value.green { color: #34d399; }
    .dash-card .value.blue { color: #818cf8; }
    .dash-card .value.purple { color: #a78bfa; }
    
    .success-box {
        background: rgba(16,185,129,0.05);
        border: 1px solid rgba(16,185,129,0.1);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .success-box .name { font-size: 16px; font-weight: 600; color: #f1f5f9; }
    .success-box .badge {
        font-size: 9px;
        background: rgba(16,185,129,0.12);
        color: #34d399;
        padding: 1px 8px;
        border-radius: 8px;
    }
    
    .db-item {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.015);
        padding: 3px 10px;
        border-radius: 4px;
        margin: 1px 0;
        border: 1px solid rgba(255,255,255,0.02);
    }
    .db-item .index { color: #475569; font-size: 10px; font-weight: 600; min-width: 24px; }
    .db-item .name { color: #e2e8f0; font-size: 12px; font-weight: 500; }
    
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        padding: 4px 0 !important;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        gap: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 14px !important;
        border-radius: 6px 6px 0 0 !important;
        font-size: 13px !important;
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE
# ============================================
st.markdown("""
<div style="
    font-size: 26px;
    font-weight: 700;
    color: #f1f5f9;
    padding: 8px 4px 4px 4px;
    margin: 0 0 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    gap: 8px;
">
    🔐 Face Recognition Pro
    <span style="font-size: 14px; font-weight: 400; color: #64748b; margin-left: 4px;">| SFace Deep Learning</span>
</div>
""", unsafe_allow_html=True)

# ============================================
# SSL + MODEL + CASCADE
# ============================================
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MODEL_PATH = "face_recognition_sface_2021dec.onnx"

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Download model
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model (35MB)... Please wait."):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Model downloaded!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            st.stop()

# Download cascade
if not os.path.exists(CASCADE_PATH):
    with st.spinner("📥 Downloading Haar Cascade..."):
        try:
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
            st.success("✅ Cascade downloaded!")
        except Exception as e:
            st.error(f"❌ Cascade download failed: {e}")
            st.stop()

@st.cache_resource
def load_recognizer():
    recognizer = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return recognizer, cascade

recognizer, face_cascade = load_recognizer()

# ============================================
# DATABASE
# ============================================
DB_FILE = "face_db.json"
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(face_db, f)

def get_face_encoding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (112, 112))
        feature = recognizer.feature(face_resized)
        return feature.flatten()
    except Exception:
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def auto_register_from_train():
    train_dir = "train"
    if not os.path.exists(train_dir) or face_db:
        return

    person_encodings = {}
    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for fname in os.listdir(person_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_folder, fname)
            feat = get_face_encoding(img_path)
            if feat is not None:
                if person_name not in person_encodings:
                    person_encodings[person_name] = []
                person_encodings[person_name].append(feat)

    for name, enc_list in person_encodings.items():
        if enc_list:
            face_db[name] = np.mean(enc_list, axis=0).tolist()

    if person_encodings:
        save_db()

def recognize_face(image_path):
    feat = get_face_encoding(image_path)
    if feat is None or not face_db:
        return None, 0.0
    best_name = None
    best_sim = -1.0
    for name, stored_vec in face_db.items():
        ref = np.array(stored_vec, dtype=np.float32)
        sim = cosine_similarity(feat, ref)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim > 0.4:
        return best_name, best_sim * 100
    return None, best_sim * 100

def register_person_manually(name, image_path):
    feat = get_face_encoding(image_path)
    if feat is None:
        return False, "No clear face found."
    face_db[name] = feat.tolist()
    save_db()
    return True, f"Registered '{name}' successfully."

try:
    auto_register_from_train()
except Exception:
    pass

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🔐 Face Pro")
    st.caption("SFace Deep Learning")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="label">👤 Registered</div>
            <div class="value">{len(face_db)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="label">🔍 Recognitions</div>
            <div class="value">{st.session_state.total_recognitions}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("#### 📜 History")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_recognitions = 0
        save_history()
        st.rerun()
    
    if not st.session_state.history:
        st.info("ℹ️ No history yet.")
    else:
        for entry in reversed(st.session_state.history[-15:]):
            acc = entry.get('accuracy', 0.0)
            name = entry.get('name', 'Unknown')
            time = entry.get('time', '--:--')
            if name == "Unknown":
                name_display = f"❌ Unknown"
                color = "#f87171"
            else:
                name_display = f"✅ {name}"
                color = "#34d399" if acc >= 70 else "#fbbf24"
            st.markdown(f"""
            <div class="history-item">
                <div class="history-name">{name_display}</div>
                <div style="display:flex; justify-content:space-between;">
                    <span class="history-time">{time}</span>
                    <span style="color:{color}; font-weight:600;">{acc:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# MAIN TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Recognize", "➕ Register", "📁 Database", "📊 Dashboard"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, width=250)
    with col2:
        if uploaded:
            with st.spinner("Analyzing..."):
                temp_path = "temp_recog.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                name, confidence = recognize_face(temp_path)
                os.remove(temp_path)
                current_time = datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append({
                    "name": name if name else "Unknown",
                    "accuracy": confidence,
                    "time": current_time
                })
                st.session_state.total_recognitions = len(st.session_state.history)
                save_history()
                if name:
                    st.markdown(f"""
                    <div class="success-box">
                        <span class="badge">✅ Match</span>
                        <div class="name">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1f}%")
                else:
                    st.warning(f"❌ Unknown Person (Confidence: {confidence:.1f}%)")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Name")
        if st.button("Register", use_container_width=True):
            if new_name and 'new_img' in locals() and new_img:
                temp_path = "temp_reg.jpg"
                with open(temp_path, "wb") as f:
                    f.write(new_img.read())
                ok, msg = register_person_manually(new_name, temp_path)
                os.remove(temp_path)
                if ok:
                    st.success(msg)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(msg)
    with col2:
        new_img = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"], key="reg")
        if new_img:
            st.image(new_img, width=250)

with tab3:
    if face_db:
        st.write(f"Total Registered: **{len(face_db)}**")
        for idx, name in enumerate(face_db.keys(), start=1):
            st.markdown(f"""
            <div class="db-item">
                <span class="index">#{idx:02d}</span>
                <span class="name">👤 {name}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Database is empty. Add images to `train` folder or register manually.")

with tab4:
    st.markdown("### 📊 System Dashboard")
    st.divider()
    
    if st.session_state.history:
        avg_acc = sum(h.get('accuracy', 0) for h in st.session_state.history) / len(st.session_state.history)
    else:
        avg_acc = 0.0
    
    model_status = "✅ Online" if os.path.exists(MODEL_PATH) else "⏳ Downloading"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="dash-card">
            <div class="icon">👤</div>
            <div class="value green">{len(face_db)}</div>
            <div class="label">Registered</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="dash-card">
            <div class="icon">🔍</div>
            <div class="value blue">{st.session_state.total_recognitions}</div>
            <div class="label">Recognitions</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="dash-card">
            <div class="icon">📊</div>
            <div class="value purple">{avg_acc:.1f}%</div>
            <div class="label">Avg Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="dash-card">
            <div class="icon">🧠</div>
            <div class="value" style="font-size:28px; color:#34d399;">{model_status}</div>
            <div class="label">Model Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### 📜 Recent Recognitions")
    if not st.session_state.history:
        st.info("ℹ️ No recognitions yet.")
    else:
        for entry in reversed(st.session_state.history[-10:]):
            acc = entry.get('accuracy', 0.0)
            name = entry.get('name', 'Unknown')
            time = entry.get('time', '--:--')
            color = "#34d399" if acc >= 70 else "#fbbf24"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; background:rgba(255,255,255,0.02); padding:6px 14px; border-radius:4px; margin:2px 0; border-left:3px solid #4f46e5;">
                <span style="color:#f1f5f9;">{name}</span>
                <span style="color:#94a3b8; font-size:12px;">{time}</span>
                <span style="color:{color}; font-weight:600;">{acc:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

st.caption("🔐 Face Recognition Pro | OpenCV SFace | Streamlit")