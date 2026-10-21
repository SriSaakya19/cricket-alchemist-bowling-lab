import base64
import hashlib
import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from dynamic_processor import (
    SUPPORTED_EXTENSIONS,
    FFmpegNotFoundError,
    ProcessorError,
    analyze_video_bytes,
    convert_uploaded_video,
    extract_video_frames,
    find_ffmpeg,
    render_3d_video,
)

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="The Cricket Alchemist | Bowling Lab",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- PREMIUM UI --------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800;900&family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
background: radial-gradient(
ellipse at 70% 0%,
#173653 0%, #071322 48%, #030711 100%
);
color: #F5F8FF;
}

.block-container {
max-width: 1600px;
padding: 1.4rem 2rem 3rem;
}

header[data-testid="stHeader"] {
background: transparent;
}

section[data-testid="stSidebar"] {
display: none;
}

.hero {
padding: 2.2rem;
border: 1px solid #345777;
border-radius: 24px;
background: linear-gradient(
120deg, #142D47, #071322 75%, #17243B
);
box-shadow: 0 12px 45px #0007;
}

.kicker {
color: #FFD16D;
letter-spacing: .19em;
font-size: .82rem;
font-weight: 800;
}

.hero h1 {
font-family: 'Barlow Condensed', sans-serif;
font-size: clamp(3rem, 7vw, 6rem);
line-height: .9;
font-weight: 900;
margin: .65rem 0 1rem;
color: #FFFFFF;
}

.gold {
color: #FFD16D;
}

.hero p {
color: #E0EAF7;
font-size: 1.05rem;
line-height: 1.75;
max-width: 950px;
