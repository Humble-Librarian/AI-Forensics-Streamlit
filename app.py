import streamlit as st
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import time
import tempfile
import io
import gdown # Added gdown for Google Drive downloads
import gc    # Added for manual memory management
import plotly.express as px
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

import config
import utils
import engine
from models import SpatialXception, SRMXception, DeepfakeLSTM

# ==========================================
#        PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    """
    Loads external CSS for custom UI styling.
    
    Args:
        file_name (str): Path to the .css file.
    """
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"Stylesheet {file_name} not found.")

load_css("styles.css")

# ==========================================
#        SESSION STATE INITIALIZATION
# ==========================================
if 'logs' not in st.session_state: st.session_state.logs = []
if 'results' not in st.session_state: st.session_state.results = None
if 'history' not in st.session_state: st.session_state.history = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'video_meta' not in st.session_state: 
    st.session_state.video_meta = {"name": "", "duration": 0, "res": "", "fps": 0}
if 'seq_length' not in st.session_state: 
    st.session_state.seq_length = config.SEQ_LENGTH

def add_log(message, type="info"):
    """
    Adds a formatted log entry to the system kernel log.
    
    Args:
        message (str): Log message.
        type (str): Status type (info, success, error).
    """
    colors = {"info": "#6366f1", "success": "#22c55e", "error": "#ef4444"}
    icon = {"info": "ℹ", "success": "✓", "error": "✖"}
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    log_entry = f"<div class='log-entry'><span style='color: {colors.get(type, '#94a3b8')}; opacity: 0.8;'>[{timestamp}]</span> <span style='color: {colors.get(type, '#94a3b8')}; font-weight: 600;'>{icon.get(type, '')} {message}</span></div>"
    st.session_state.logs.append(log_entry)
    if len(st.session_state.logs) > 30:
        st.session_state.logs.pop(0)

# ==========================================
#        MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Initializing Tensor Cores & Pulling Models from Cloud...")
def boot_system():
    """
    Caches and initializes the neural inference engine.
    
    Returns:
        tuple: (Loaded models, device)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = engine.load_all_models(device, log_func=add_log)
    return models, device

models, DEVICE = boot_system()

# ==========================================
#        INFERENCE ENGINE
# ==========================================
def run_analysis(video_path, seq_length):
    """
    UI-integrated wrapper for running the forensic analysis pipeline.
    
    Args:
        video_path (str): Path to analyzed video.
        seq_length (int): Number of frames for analysis.
    """
    add_log(">>> INITIATING DEEP FORENSIC SCAN...", "info")
    
    if models is None:
        add_log("SYSTEM ERROR: Neural Engine Offline", "error")
        return

    # Progress bar for the UI
    progress_bar = st.progress(0)
    
    def ui_progress(pct):
        progress_bar.progress(pct)

    results = engine.run_inference(
        video_path, 
        seq_length, 
        models, 
        DEVICE, 
        log_func=add_log,
        progress_callback=ui_progress
    )
    
    if results:
        st.session_state.results = results
        st.session_state.history.insert(0, results)
        progress_bar.empty()
    else:
        add_log("Inference failed to produce results.", "error")

# ==========================================
# ==========================================

# --- LEFT SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
            <div style='background: #4f46e5; border-radius: 50%; width: 24px; height: 28px; border-bottom-right-radius: 0;'></div>
            <h2 style='margin: 0; font-size: 20px; font-weight: 900; letter-spacing: 0.5px;'>{config.APP_TITLE.upper()}</h2>
        </div>
        <p style='color: #4b5563; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-bottom: 30px; margin-top: -5px;'>CONSOLE v{config.VERSION}</p>
    """, unsafe_allow_html=True)
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "⊞ DASHBOARD"

    st.markdown("<div class='nav-header'>MAIN</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        options=["⊞ DASHBOARD", "⏱ CASE HISTORY", "⚙ SETTINGS"],
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state.current_page = page
    
    st.markdown("<br><div class='nav-header' style='margin-top: 30px;'>SYSTEM</div>", unsafe_allow_html=True)
    
    loaded_color = "#22c55e" if models is not None else "#ef4444"
    loaded_text = "Ready" if models is not None else "Error"
    gpu_color = "#22c55e" if torch.cuda.is_available() else "#9ca3af"
    gpu_text = "Active" if torch.cuda.is_available() else "N/A"
    
    st.markdown(f"""
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
            <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 6px; height: 6px; border-radius: 50%; background-color: {loaded_color};'></div><span style='color: #9ca3af; font-size: 12px;'>Model Loaded</span></div>
            <span style='color: {loaded_color}; font-size: 12px; font-weight: 600;'>{loaded_text}</span>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
            <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 6px; height: 6px; border-radius: 50%; background-color: {gpu_color};'></div><span style='color: #9ca3af; font-size: 12px;'>GPU Acceleration</span></div>
            <span style='color: #9ca3af; font-size: 12px; font-weight: 600;'>{gpu_text}</span>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
            <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 6px; height: 6px; border-radius: 50%; background-color: #22c55e;'></div><span style='color: #9ca3af; font-size: 12px;'>System Version</span></div>
            <span style='color: #d1d5db; font-size: 12px; font-weight: 600;'>{config.VERSION}</span>
        </div>
    """, unsafe_allow_html=True)
    
if st.session_state.current_page == "⊞ DASHBOARD":
    col1, col_gap, col2 = st.columns([1.5, 0.05, 1])

    # --- CENTER PANEL (Video, Meta, Frame Chart, Logs) ---
    with col1:
        uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], label_visibility="collapsed")
    
        if uploaded_file is not None:
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_identifier or st.session_state.video_path is None or not os.path.exists(st.session_state.video_path):
                uploaded_file.seek(0)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                tfile.write(uploaded_file.read())
                tfile.close()
                st.session_state.video_path = tfile.name
                st.session_state.results = None
                st.session_state.current_file_id = file_identifier
                st.session_state.video_meta = utils.get_video_metadata(tfile.name, uploaded_file.name)
                st.rerun()
            
        if st.session_state.video_path:
            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
            st.video(st.session_state.video_path)
            st.markdown("</div>", unsafe_allow_html=True)
        
            m = st.session_state.video_meta
            st.markdown(f"""
                <div class='metadata-row'>
                    <div class='meta-pill'>📄 {m['name']}</div>
                    <div class='meta-pill'>⏱ {m['duration']}</div>
                    <div class='meta-pill'>📐 {m['res']}</div>
                    <div class='meta-pill'>🎞 {m['fps']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='video-container' style='height: 360px; align-items: center; border: 1px dashed #374151;'>
                    <p style='color: #64748b; font-weight: 600;'>DRAG & DROP MEDIA HERE OR CLICK ABOVE</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='frame-analysis-box'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title' style='margin-top: 0;'>FRAME-BY-FRAME ANALYSIS</div>", unsafe_allow_html=True)
    
        if st.session_state.results:
            scores = st.session_state.results['frame_scores']
            blocks_html = "<div class='frame-blocks-row'>"
            for score in scores:
                c = "fk-red" if score > 0.6 else ("fk-yellow" if score > 0.35 else "fk-green")
                blocks_html += f"<div class='frame-block {c}'></div>"
            blocks_html += "</div>"
            st.markdown(blocks_html, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='frame-blocks-row'>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                    <div class='frame-block' style='background-color: #272733;'></div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
        st.markdown("<div class='log-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title' style='margin-top: 0;'>SYSTEM KERNEL LOG</div>", unsafe_allow_html=True)
        log_html = "".join(st.session_state.logs)
        st.markdown(f"<div class='log-box'>{log_html if log_html else '<span style=\"color:#4b5563;\">SYSTEM STANDBY...</span>'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- RIGHT PANEL (Metrics & Actions) ---
    with col2:
        st.markdown("<div class='metrics-panel'>", unsafe_allow_html=True)
    
        st.markdown("<div class='manipulation-label'>MANIPULATION PROBABILITY</div>", unsafe_allow_html=True)
    
        res = st.session_state.results
        if res:
            final = res['final']
            if final > 0.6:
                color_class, verdict, display_score = "score-red", "MANIPULATED", final*100
                badge_class = "status-red"
            elif final > 0.35:
                color_class, verdict, display_score = "score-yellow", "SUSPICIOUS", final*100
                badge_class = "status-yellow"
            else:
                color_class, verdict, display_score = "score-green", "AUTHENTIC", (1-final)*100
                badge_class = "status-green"
            
            st.markdown(f"<div class='score-value {color_class}'>{display_score:.0f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='status-badge {badge_class}'>{verdict}</div>", unsafe_allow_html=True)
        
            st.markdown("<div class='metric-title'>DETAILED METRICS</div>", unsafe_allow_html=True)
        
            def render_metric(label, val):
                bar_color = "#6366f1"
                st.markdown(f"""
                    <div class='metric-title' style='margin-top: 20px; margin-bottom: 5px; color: #d1d5db; font-size: 9px;'>{label}</div>
                    <div class='progress-row'>
                        <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {val*100}%; background-color: {bar_color};'></div></div>
                        <div class='progress-value'>{val*100:.0f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            render_metric("SPATIAL ARTIFACTS", res['s'])
            render_metric("FREQUENCY NOISE", res['f'])
            render_metric("TEMPORAL CONSISTENCY", res['t'])
        
        else:
            st.markdown(f"<div class='score-value score-red'>--%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='status-badge' style='background: rgba(255,255,255,0.05); color: #64748b; border: 1px solid rgba(255,255,255,0.1);'>AWAITING DATA</div>", unsafe_allow_html=True)
        
            st.markdown("<div class='metric-title'>DETAILED METRICS</div>", unsafe_allow_html=True)
            for label in ["SPATIAL ARTIFACTS", "FREQUENCY NOISE", "TEMPORAL CONSISTENCY"]:
                st.markdown(f"""
                    <div class='metric-title' style='margin-top: 20px; margin-bottom: 5px; color: #64748b; font-size: 9px;'>{label}</div>
                    <div class='progress-row'>
                        <div class='progress-bar-bg'></div>
                        <div class='progress-value' style='color: #64748b;'>--%</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='flex-grow: 1; min-height: 40px;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-title' style='font-size: 9px;'>SEQUENCE RESOLUTION (10-25)</div>", unsafe_allow_html=True)
        st.session_state.seq_length = st.slider("Sequence Length", min_value=10, max_value=25, value=st.session_state.seq_length, step=1, label_visibility="collapsed")

        if st.button("▶ RUN DIAGNOSTICS", type="primary"):
            if st.session_state.video_path:
                st.session_state.results = None
                st.session_state.logs = []
                run_analysis(st.session_state.video_path, st.session_state.seq_length)
                st.rerun()
            else:
                add_log("No media file provided.", "error")
            
        if res:
            pdf_buffer = utils.generate_pdf_buffer(res, st.session_state.video_path)
            st.download_button(
                label="⬇ EXPORT REPORT",
                data=pdf_buffer,
                file_name=f"Report_{res['filename']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary"
            )
        else:
            st.button("⬇ EXPORT REPORT", type="secondary", disabled=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == '⏱ CASE HISTORY':
    st.markdown("<div class='metrics-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title' style='font-size: 16px; margin-top:0;'>⏱ ANALYSIS HISTORY</div>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.markdown("""
            <div style='background: #181820; border: 1px dashed #374151; border-radius: 8px; padding: 40px; text-align: center; margin-top: 20px;'>
                <h3 style='color: #64748b; margin: 0;'>NO PREVIOUS CASES</h3>
                <p style='color: #4b5563; font-size: 12px; margin-top: 8px;'>Run an analysis from the dashboard to see history here.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for idx, res in enumerate(st.session_state.history):
            final = res['final']
            if final > 0.6: badge, color = "MANIPULATED", "#ef4444"
            elif final > 0.35: badge, color = "SUSPICIOUS", "#eab308"
            else: badge, color = "AUTHENTIC", "#22c55e"
            
            st.markdown(f"""
                <div style='background: #0b0b0f; border: 1px solid #272733; border-radius: 8px; padding: 16px; margin-top: 16px;'>
                    <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                        <div>
                            <h4 style='color: #e5e7eb; margin: 0;'>{res['filename']}</h4>
                            <p style='color: #64748b; font-size: 11px; margin-top: 4px; margin-bottom: 0;'>{res.get('timestamp', 'Unknown Time')} &bull; {res.get('seq_length', 'N/A')} frames</p>
                        </div>
                        <div style='text-align: right;'>
                            <div style='color: {color}; font-size: 20px; font-weight: 800;'>{final*100:.1f}%</div>
                            <div style='color: {color}; font-size: 10px; font-weight: 700; letter-spacing: 1px;'>{badge}</div>
                        </div>
                    </div>
                    <div style='display: flex; gap: 16px; margin-top: 16px; border-top: 1px solid #1f1f2e; padding-top: 12px;'>
                        <div style='flex: 1;'>
                            <div style='color: #9ca3af; font-size: 9px; font-weight: 700;'>SPATIAL</div>
                            <div style='color: #d1d5db; font-size: 12px; font-weight: 600;'>{res['s']*100:.0f}%</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #9ca3af; font-size: 9px; font-weight: 700;'>FREQUENCY</div>
                            <div style='color: #d1d5db; font-size: 12px; font-weight: 600;'>{res['f']*100:.0f}%</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #9ca3af; font-size: 9px; font-weight: 700;'>TEMPORAL</div>
                            <div style='color: #d1d5db; font-size: 12px; font-weight: 600;'>{res['t']*100:.0f}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        if st.button("🗑 CLEAR HISTORY", type="secondary", key="clear_history_btn"):
            st.session_state.history = []
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == '⚙ SETTINGS':
    st.markdown("<div class='metrics-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title' style='font-size: 16px; margin-top:0;'>⚙ SYSTEM SETTINGS</div>", unsafe_allow_html=True)

    colA, col_gap_set, colB = st.columns([1, 0.1, 1])

    with colA:
        st.markdown("<div class='metric-title'>CORE PARAMETERS</div>", unsafe_allow_html=True)
        new_seq = st.slider("Global Sequence Length (Frames per Analysis)", min_value=10, max_value=25, value=st.session_state.seq_length, step=1)
        if new_seq != st.session_state.seq_length:
            st.session_state.seq_length = new_seq
            st.rerun()

        st.markdown("<div class='metric-title' style='margin-top: 30px;'>SYSTEM ACTIONS</div>", unsafe_allow_html=True)
        if st.button("🔄 REBOOT NEURAL ENGINE", type="secondary", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
            
        if st.button("🗑 CLEAR SYSTEM LOGS", type="secondary", use_container_width=True):
            st.session_state.logs = []
            st.rerun()

    with colB:
        st.markdown("<div class='metric-title'>DIAGNOSTICS & HARDWARE</div>", unsafe_allow_html=True)
        
        status_color = "#22c55e" if torch.cuda.is_available() else "#eab308"
        device_nm = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.markdown(f"""
            <div style='background: #0b0b0f; border: 1px solid #272733; border-radius: 8px; padding: 16px; margin-bottom: 16px;'>
                <div style='color: #9ca3af; font-size: 10px; font-weight: 700; margin-bottom: 4px;'>COMPUTE DEVICE</div>
                <div style='color: {status_color}; font-size: 14px; font-weight: 600;'>{device_nm}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style='background: #0b0b0f; border: 1px solid #272733; border-radius: 8px; padding: 16px;'>
                <div style='color: #9ca3af; font-size: 10px; font-weight: 700; margin-bottom: 12px;'>MODEL WEIGHTS</div>
        """, unsafe_allow_html=True)
        
        for m_name, path in config.MODEL_FILES.items():
            exists = os.path.exists(path)
            color = "#22c55e" if exists else "#ef4444"
            icon = "✓" if exists else "✖"
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #1f1f2e; padding-bottom: 8px;'>
                    <span style='color: #d1d5db; font-size: 12px; font-weight: 600;'>{m_name.upper()}</span>
                    <span style='color: {color}; font-size: 12px; font-weight: 800;'>{icon}</span>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
