import streamlit as st
import os
import time
import cv2
import joblib
import numpy as np
import tensorflow as tf
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# Make the app use the full browser width and reduce default padding to fit Windows screen
st.set_page_config(page_title="Milk Classifier", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      /* overall theme: black background, cyan highlights */
      html, body, #root, .appview-container, .main, .reportview-container .main .block-container {
          background: #000000 !important;
          color: #00ffff !important; /* cyan text */
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
          color: #00ffff;
          border: 1px solid #00cccc;
          cursor: pointer;
          font-weight: 600;
      }

      /* Sidebar styling */
      .css-1d391kg { background-color: #041111 !important; } /* darker cyan tint */
      .css-1y0tads { color: #00ffff !important; } /* cyan headings */

      /* make markdown text use cyan tint */
      .stMarkdown, .stText, .stInfo {
          color: #00ffff !important;
      }

      /* Center prediction badge */
      .prediction-center {
          display: flex;
          justify-content: center;
          align-items: center;
          margin: 20px 0;
      }

      /* Custom positioning for upload section */
      .upload-prediction-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        margin: 20px auto;
        position: relative;
        right: 150px;  /* Move left by changing this value */
    }
    
    .upload-image-container {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .upload-image-container img {
        max-width: 25% !important;
        height: auto !important;
    }
    
    /* Prediction badge styling for upload section */
    .prediction-badge-upload {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 18px;
        text-align: center;
        min-width: 160px;
        margin-top: 10px;
        position: relative;
        left: 335px;  /* Move prediction badge left */
    }

    /* Confidence section styling for upload */
    .confidence-container {
        width: 100%;
        max-width: 300px;    /* Control width of confidence section */
        margin: 10px auto;   /* Center and add spacing */
        text-align: center;
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
def load_model():
    """Load the trained CNN model"""
    model_path = os.path.join('models', 'milk_classifier_cnn.keras')
    return tf.keras.models.load_model(model_path)

# Update model loading
model = load_model()
labels = {0: 'Pure', 1: 'Glucose', 2: 'Adulterated', 3: 'Pathogens'}

def preprocess_image(image_path):
    """Preprocess image for CNN prediction"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype('float32') / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def predict(image_path):
    """Predict using CNN model"""
    features = preprocess_image(image_path)
    predictions = model.predict(features)
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])
    label = labels[pred_idx]
    return label, confidence

# helper: colored badge for predictions (keep existing colors)
def render_prediction_badge(text, center=True):
    colors = {
        "Adulterated": "#ff4b4b",  # red
        "Pure": "#00e676",         # green
        "Pathogens": "#9c27b0",    # purple
        "Glucose": "#ffffff"       # white
    }
    bg = colors.get(text, "#666666")
    txt_color = "#000000" if bg == "#ffffff" else "#ffffff"
    badge_html = f"""
    <div style="
        display:inline-block;
        padding:10px 16px;
        border-radius:10px;
        background:{bg};
        color:{txt_color};
        font-weight:700;
        font-size:18px;
        text-align:center;
        min-width:160px;
    ">{text}</div>
    """
    if center:
        html = f'<div class="prediction-center">{badge_html}</div>'
    else:
        html = badge_html
    st.markdown(html, unsafe_allow_html=True)

# sidebar debug panel (separate node)
with st.sidebar.expander("Debug / Diagnostics", expanded=True):
    # smaller font for cleaner compact display
    st.markdown('<div style="font-size:12px">', unsafe_allow_html=True)
    st.write("Last checked:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("CAPTURE_FOLDER:", CAPTURE_FOLDER)
    try:
        dbg_files = sorted(
            [f for f in os.listdir(CAPTURE_FOLDER) if os.path.isfile(os.path.join(CAPTURE_FOLDER, f))],
            key=lambda x: os.path.getmtime(os.path.join(CAPTURE_FOLDER, x)), reverse=True
        )
        st.write("files (newest→oldest):")
        for f in dbg_files[:10]:
            st.markdown(f"- {f}", unsafe_allow_html=True)
    except Exception as e:
        st.write("ERR listing files:", e)
    st.write("session_state.last_file:", st.session_state.get("last_file"))
    st.write("history length:", len(st.session_state.get("history", [])))
    if st.button("Force refresh"):
        # quick way to force rerun from sidebar
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.experimental_set_query_params(_refresh=str(time.time()))
    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar.expander("Manage Captures", expanded=False):
    st.markdown('<div style="font-size:12px">', unsafe_allow_html=True)
    st.write("Safe Delete")
    # Prepare-delete preview (existing behavior)
    if st.button("Prepare delete", key="prepare_delete"):
        try:
            names = sorted(os.listdir(CAPTURE_FOLDER))
            deletables = []
            for name in names:
                path = os.path.join(CAPTURE_FOLDER, name)
                # skip directories (including 'thumbnail') to be safe
                if os.path.isdir(path):
                    continue
                # only include common image types
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    deletables.append(name)
            st.session_state._delete_preview = deletables
        except Exception as e:
            st.error(f"Error preparing delete: {e}")
            st.session_state._delete_preview = []

    if st.session_state.get("_delete_preview"):
        preview = st.session_state["_delete_preview"]
        st.write(f"Files to delete ({len(preview)}):")
        # truncated listing
        for fn in preview[:50]:
            st.markdown(f"- {fn}", unsafe_allow_html=True)
        if len(preview) > 50:
            st.write("... (truncated)")
        if st.button("Confirm delete", key="confirm_delete"):
            deleted = 0
            errors = []
            for name in preview:
                p = os.path.join(CAPTURE_FOLDER, name)
                try:
                    os.remove(p)
                    deleted += 1
                except Exception as e:
                    errors.append((name, str(e)))
            st.success(f"Deleted {deleted} files.")
            if errors:
                st.error(f"Errors deleting {len(errors)} files.")
                for n, err in errors[:10]:
                    st.write(f"- {n}: {err}")
            # clear preview and reset session markers
            st.session_state._delete_preview = []
            st.session_state.last_file = None
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.experimental_set_query_params(_refresh=str(time.time()))

    st.markdown("---")
    # Immediate delete-all action (skips directories)
    st.write("Force Delete")
    if st.button("Delete all images", key="delete_all_images"):
        deleted = 0
        errors = []
        try:
            for name in os.listdir(CAPTURE_FOLDER):
                p = os.path.join(CAPTURE_FOLDER, name)
                if os.path.isdir(p):
                    continue
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    try:
                        os.remove(p)
                        deleted += 1
                    except Exception as e:
                        errors.append((name, str(e)))
            st.success(f"Deleted {deleted} files.")
            if errors:
                st.error(f"Errors deleting {len(errors)} files.")
        except Exception as e:
            st.error(f"Failed to delete files: {e}")
        # reset state and refresh UI
        st.session_state.last_file = None
        st.session_state.history = []
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.experimental_set_query_params(_refresh=str(time))
    st.markdown('</div>', unsafe_allow_html=True)
#
st.title("🔬 Real-time Milk Sample Classification")
st.write("Waiting for new captures in CAPTURE_FOLDER...")

# Add show_conf definition here
show_conf = st.sidebar.checkbox("Show confidence values", value=False, key="show_conf_main")

# Update the file upload section with simpler code
st.markdown("### 📤 Upload Image for Classification")
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])

# Update the prediction section in the upload handler
if uploaded_file is not None:
    try:
        # Read file directly from memory
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Preprocessing (without debug prints)
        img = cv2.resize(image, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])
        pred_label = labels[pred_idx]
        
        # Updated display section with better styling
        st.markdown('<div class="upload-prediction-container">', unsafe_allow_html=True)
        
        # Image container
        st.markdown('<div class="upload-image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Custom prediction badge
        colors = {
            "Adulterated": "#ff4b4b",
            "Pure": "#00e676",
            "Pathogens": "#9c27b0",
            "Glucose": "#ffffff"
        }
        bg = colors.get(pred_label, "#666666")
        txt_color = "#000000" if bg == "#ffffff" else "#ffffff"
        
        st.markdown(
            f"""<div class="prediction-badge-upload" style="background:{bg};color:{txt_color};">
                {pred_label}
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Show confidence if enabled
        if show_conf:
            st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
            conf_pct = int(confidence * 100)
            st.markdown(f'Confidence: {conf_pct}%', unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing image: {e}")

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
        pred_label, pred_conf = predict(fullpath)
    except Exception as e:
        pred_label, pred_conf = f"ERROR: {e}", 0.0
    st.session_state.history.append({
        "file": fullpath,
        "time": datetime.fromtimestamp(os.path.getmtime(fullpath)).strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": pred_label,
        "confidence": pred_conf
    })
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
        st.image(latest["file"], caption=f"{os.path.basename(latest['file'])} — {latest['time']}", use_container_width=False)
        
        # Show centered prediction badge
        render_prediction_badge(latest["prediction"], center=True)
        
        # Show confidence below prediction
        if "confidence" in latest:
            conf_pct = int(latest["confidence"] * 100)
            st.markdown(
                f"""<div style="text-align: center; color: #00ffff;">
                    Confidence: {conf_pct}%
                </div>""", 
                unsafe_allow_html=True
            )
            # Centered progress bar with custom width
            col_left, col_middle, col_right = st.columns([1, 2, 1])
            with col_middle:
                st.progress(latest["confidence"])
    else:
        st.info("No captures classified yet.")

with col2:
    st.markdown("### 📊 Analysis Dashboard")
    
    # Statistics card
    with st.container():
        st.markdown("""
        <div style="
            background-color: rgba(0,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #00ffff;
        ">
        """, unsafe_allow_html=True)
        
        if st.session_state.history:
            # Today's statistics
            today = datetime.now().date()
            today_samples = sum(1 for x in st.session_state.history 
                              if datetime.strptime(x['time'], "%Y-%m-%d %H:%M:%S").date() == today)
            
            # Category counts
            total = len(st.session_state.history)
            pure_count = sum(1 for x in st.session_state.history if x['prediction'] == 'Pure')
            adulterated_count = sum(1 for x in st.session_state.history if x['prediction'] == 'Adulterated')
            glucose_count = sum(1 for x in st.session_state.history if x['prediction'] == 'Glucose')
            pathogen_count = sum(1 for x in st.session_state.history if x['prediction'] == 'Pathogens')
            
            # Display stats
            st.markdown("#### Today's Analysis")
            st.markdown(f"📊 Samples analyzed today: **{today_samples}**")
            st.markdown("#### Overall Statistics")
            st.markdown(f"🔍 Total samples: **{total}**")
            
            # Category breakdown with colored indicators
            st.markdown("""
            <style>
            .stat-row {
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 5px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            def make_stat_row(label, count, color):
                percentage = (count/total)*100 if total > 0 else 0
                return f"""
                <div class="stat-row" style="background-color: {color}22;">
                    <span>{label}:</span>
                    <span><b>{count}</b> ({percentage:.1f}%)</span>
                </div>
                """
            
            stats_html = f"""
            {make_stat_row("Pure", pure_count, "#00e676")}
            {make_stat_row("Adulterated", adulterated_count, "#ff4b4b")}
            {make_stat_row("Glucose", glucose_count, "#ffffff")}
            {make_stat_row("Pathogens", pathogen_count, "#9c27b0")}
            """
            st.markdown(stats_html, unsafe_allow_html=True)
            
            # Recent detections timeline
            st.markdown("#### Recent Timeline")
            recent = list(reversed(st.session_state.history[-5:]))  # last 5 entries
            for item in recent:
                st.markdown(f"""
                <div style="
                    padding: 5px;
                    margin: 5px 0;
                    font-size: 0.9em;
                    border-left: 3px solid #00ffff;
                    padding-left: 10px;
                ">
                    <div>{item['time']}</div>
                    <div><b>{item['prediction']}</b></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("No data available yet")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Optional: Add export buttons
    st.markdown("### 📥 Export Data")
    if st.session_state.history:
        if st.button("Export to CSV"):
            # Prepare CSV data
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Timestamp', 'Prediction', 'Confidence', 'Filename'])
            
            for item in st.session_state.history:
                writer.writerow([
                    item['time'],
                    item['prediction'],
                    f"{int(item.get('confidence', 0)*100)}%",
                    os.path.basename(item['file'])
                ])
            
            st.download_button(
                label="Download CSV",
                data=output.getvalue(),
                file_name=f"milk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

# Remove the history from col2 and add to sidebar (place after other sidebar expanders)
with st.sidebar.expander("History", expanded=False):
    st.markdown('<div style="font-size:12px">', unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown('<div class="history-scroll">', unsafe_allow_html=True)
        for item in reversed(st.session_state.history):
            with st.container():
                try:
                    st.image(item["file"], width=100)
                except Exception:
                    st.write("")
                st.markdown(f"**{os.path.basename(item['file'])}**")
                st.write(item["time"])
                render_prediction_badge(item["prediction"], center=False)
                if "confidence" in item:
                    conf_pct = int(item["confidence"] * 100)
                    st.progress(item["confidence"])
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No history entries.")
    st.markdown('</div>', unsafe_allow_html=True)

# Remove the duplicate show_conf checkbox from the sidebar expander (around line 450)
with st.sidebar.expander("Prediction Confidence", expanded=False):
    st.markdown('<div style="font-size:12px">', unsafe_allow_html=True)
    if st.session_state.history and show_conf:
        latest = st.session_state.history[-1]
        if "confidence" in latest:
            conf_pct = int(latest["confidence"] * 100)
            st.write(f"Latest prediction confidence: {conf_pct}%")
            st.progress(latest["confidence"])
    st.markdown('</div>', unsafe_allow_html=True)

# Add after the existing statistics in the Analysis Dashboard section
# Around line 530, inside the statistics card container

# ...existing statistics code...

# Add model accuracy section
st.markdown("#### Model Performance")
st.markdown("""
<div class="stat-row" style="background-color: #00ffff22;">
    <span>Model Accuracy:</span>
    <span><b>86.81%</b></span>
</div>
""", unsafe_allow_html=True)

# Add more details if show_conf is enabled
if show_conf:
    st.markdown("""
    <div style="font-size: 0.9em; margin-top: 5px;">
        <ul style="margin-left: 20px; color: #00ffff;">
            <li>Training Accuracy: 99.13%</li>
            <li>Validation Accuracy: 86.81%</li>
            <li>Total Parameters: 17.1M</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ...continue with existing code...
