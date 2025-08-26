<!-- [STEP ONE:]

{ ACTIVATE VIRTUAL ENVIRONMENT}

().\venv\Scripts\activate)

 -->
  
<!-- [STEP TWO:] 
{HOW TO CHECK THE PROJECT:}

(python scripts/predict.py)

NOTE: OUTPUT
📂 Enter path to test image: (data/raw/simulated/pure/pure_01.jpg)  GIVE YOUR OWN PATH SO THAT IT CAN PRECDICT
🧠 Prediction: Pure
-->


<!-- BUG:{IF IT DOESNT WORK START BY TRAINING THE MODEL AGAIN}

(python scripts/train_model.py)

 -->




<!-- {TO DISPLAY THE INTERFACE} 

(streamlit run app/app.py)

: -->


import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load('models/milk_classifier.pkl')

# Class labels — must match training order!
labels = {
    0: 'Adulterated',
    1: 'Glucose',
    2: 'Pathogens',
    3: 'Pure'
}

# Preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return img_array.flatten().reshape(1, -1)

# Streamlit UI
st.title("🥛 Milk Adulteration Detector")
st.markdown("Upload a microscopic image of a milk sample to detect adulteration.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = preprocess_image(image)
    prediction = model.predict(features)[0]

    st.markdown(f"### 🧠 Prediction: `{labels[prediction]}`")



## 2
import streamlit as st
import os
import time
import cv2
import joblib
import numpy as np
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# Make the app use the full browser width and reduce default padding to fit Windows screen
st.set_page_config(page_title="Milk Classifier", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      /* force app to exactly match viewport and hide global scrollbar */
      html, body, #root, .appview-container, .main, .reportview-container .main .block-container {
          height: 100vh !important;
          overflow: hidden;
      }

      /* reduce Streamlit page padding */
      .reportview-container .main .block-container{
          padding-top:6px;
          padding-right:8px;
          padding-left:8px;
      }

      /* make columns container fill remaining height so inner pieces can scroll independently */
      div[data-testid="stColumns"] {
          height: calc(100vh - 56px);
      }

      /* make all Streamlit images display at ~25% of their container (1/4 size) */
      div[data-testid="stImage"] img {
          max-width: 25% !important;
          height: auto !important;
          object-fit: contain !important;
          display: block;
          margin-left: auto;
          margin-right: auto;
      }

      /* ensure large images don't exceed available height */
      div[data-testid="stImage"] img {
          max-height: calc(100vh - 140px) !important;
      }

      /* scrollable history area within the page (prevents whole-page scroll) */
      .history-scroll {
          max-height: calc(100vh - 140px);
          overflow: auto;
          padding-right: 6px;
      }

      /* small style for the fullscreen button */
      .fullscreen-btn {
          position: fixed;
          top: 10px;
          right: 10px;
          z-index: 9999;
          padding: 8px 12px;
          border-radius: 6px;
          background: #0e1117;
          color: white;
          border: none;
          cursor: pointer;
          font-weight: 600;
      }
    </style>

    <!-- Fullscreen toggle: requests browser fullscreen. Users can also press F11 -->
    <button class="fullscreen-btn" onclick="(function(){
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen && document.exitFullscreen();
        }
    })()">Enter / Exit Fullscreen</button>
    """,
    unsafe_allow_html=True,
)

# Paths
CAPTURE_FOLDER = "data/raw/real/CAPTURE_FOLDER"

# Load model (cached so reruns are cheap)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model("models/milk_classifier.pkl")
labels = {0: 'Adulterated', 1: 'Glucose', 2: 'Pathogens', 3: 'Pure'}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img.flatten().reshape(1, -1)

def predict(image_path):
    features = preprocess_image(image_path)
    prediction = model.predict(features)[0]
    return labels[prediction]

st.title("🔬 Real-time Milk Sample Classification")
st.write("Waiting for new captures in CAPTURE_FOLDER...")

# Auto-refresh in browser every 2000 ms (2s)
st_autorefresh(interval=2000, key="autorefresh")

# Initialize persistent UI state
if "last_file" not in st.session_state:
    st.session_state.last_file = None
if "history" not in st.session_state:
    # history: list of dicts {file, timestamp, prediction}
    st.session_state.history = []

# Safe listing of files (recomputed each rerun)
try:
    all_files = sorted(
        [f for f in os.listdir(CAPTURE_FOLDER) if os.path.isfile(os.path.join(CAPTURE_FOLDER, f))],
        key=lambda x: os.path.getmtime(os.path.join(CAPTURE_FOLDER, x)), reverse=True
    )
except Exception:
    st.error(f"Cannot access capture folder: {CAPTURE_FOLDER}")
    st.stop()

# Determine which new files to process (those newer than last_file)
to_process = []
if all_files:
    # If first run, only process the latest file (avoid backlog)
    if st.session_state.last_file is None:
        to_process = [all_files[0]]
    else:
        # collect files until we reach last_file (all_files sorted newest->oldest)
        for fname in all_files:
            full = os.path.join(CAPTURE_FOLDER, fname)
            if full == st.session_state.last_file:
                break
            to_process.append(fname)
    # process in chronological order (oldest first)
    to_process = list(reversed(to_process))

# Helper to check file stability (small wait to avoid reading partial writes)
def is_stable(path, wait=0.2):
    try:
        s1 = os.path.getsize(path)
        time.sleep(wait)
        s2 = os.path.getsize(path)
        return s1 == s2 and s1 > 0
    except Exception:
        return False

# Process new files
for fname in to_process:
    fullpath = os.path.join(CAPTURE_FOLDER, fname)
    if not is_stable(fullpath):
        # skip unstable file; it'll be picked up on next refresh
        continue
    try:
        pred = predict(fullpath)
    except Exception as e:
        pred = f"ERROR: {e}"
    st.session_state.history.append({
        "file": fullpath,
        "time": datetime.fromtimestamp(os.path.getmtime(fullpath)).strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": pred
    })
    # update last_file to the newest processed
    st.session_state.last_file = fullpath

# If there are files but none processed (e.g., only unstable), ensure last_file points to latest to avoid backlog
if all_files and not to_process and st.session_state.last_file is None:
    st.session_state.last_file = os.path.join(CAPTURE_FOLDER, all_files[0])

# UI: show latest image and history
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("Latest Capture")
        # show reduced-size image (CSS forces ~25%); disable use_container_width so CSS controls sizing
        st.image(latest["file"], caption=f"{os.path.basename(latest['file'])} — {latest['time']}", use_container_width=False)
        st.success(f"🧠 Prediction: **{latest['prediction']}**")
    else:
        st.info("No captures classified yet.")

with col2:
    st.subheader("History")
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.last_file = None
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.experimental_set_query_params(_cleared=str(time.time()))
    # show history inside an expander to save vertical space and make internal scrolling
    with st.expander("Show history", expanded=False):
        if st.session_state.history:
            # wrap history items in a scrollable div so page height stays fixed
            st.markdown('<div class="history-scroll">', unsafe_allow_html=True)
            # Show most recent first
            for item in reversed(st.session_state.history):
                row = st.container()
                with row:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        try:
                            st.image(item["file"], width=120)
                        except Exception:
                            st.write("")  # keep layout if image fails
                    with c2:
                        st.markdown(f"**{os.path.basename(item['file'])}**")
                        st.write(item["time"])
                        st.write(f"**{item['prediction']}**")
                        st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("No history entries.")
