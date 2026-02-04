import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
from color_utils import get_face_data, analyze_12_seasons, apply_drape, PALETTES, AVOID_PALETTES

st.set_page_config(page_title="Color Persona Studio", layout="wide")

# Styling for visual buttons and boxes
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 50px; border-radius: 5px; border: 1px solid #ddd; }
    .color-block { width: 100%; height: 80px; border-radius: 8px; border: 1px solid #ddd; }
    .avoid-block { width: 100%; height: 80px; border-radius: 8px; border: 1px solid #ddd; position: relative; background-color: #eee; }
    .avoid-block::after { content: '✕'; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 30px; color: rgba(255,0,0,0.5); }
    </style>
""", unsafe_allow_html=True)

def create_pdf(season, best_colors, avoid_colors, hex_skin):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(200, 20, "Personal Color Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 15, f"Your Season: {season}", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Detected Skin Tone: {hex_skin}", ln=True, align='C')
    
    pdf.ln(10)
    pdf.cell(200, 10, "Best Harmonious Colors:", ln=True)
    for color in best_colors:
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_fill_color(r, g, b)
        pdf.rect(pdf.get_x(), pdf.get_y(), 40, 10, 'F')
        pdf.set_x(pdf.get_x() + 45)
    
    pdf.ln(20)
    pdf.cell(200, 10, "Colors to Avoid:", ln=True)
    for color in avoid_colors:
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_fill_color(r, g, b)
        pdf.rect(pdf.get_x(), pdf.get_y(), 40, 10, 'F')
        pdf.set_x(pdf.get_x() + 45)
        
    return pdf.output()

st.title("🎨 Personal Color Studio")
uploaded_file = st.sidebar.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])

if "drape_color" not in st.session_state:
    st.session_state.drape_color = "#FFFFFF"

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    avg_rgb, processed_img, landmarks = get_face_data(image)
    
    if avg_rgb is not None:
        season = analyze_12_seasons(avg_rgb)
        best_p = PALETTES[season]
        avoid_p = AVOID_PALETTES[season]
        hex_skin = '#%02x%02x%02x' % tuple(avg_rgb)

        col_left, col_right = st.columns([2, 1])

        with col_right:
            st.subheader("Analysis")
            st.info(f"You are a **{season}**")
            st.markdown(f"**Sampled Skin:** <div class='color-block' style='background-color:{hex_skin};'></div>", unsafe_allow_html=True)
            
            st.write("---")
            st.write("**Quick Drape Test**")
            # Create a grid of color buttons
            grid = st.columns(4)
            for idx, color in enumerate(best_p + avoid_p):
                if grid[idx % 4].button(" ", key=f"btn_{color}", help=f"Test {color}"):
                    st.session_state.drape_color = color
                # Add a visual indicator of the button color
                grid[idx % 4].markdown(f"<div style='background-color:{color}; height:10px; border-radius:2px;'></div>", unsafe_allow_html=True)

            st.write("---")
            pdf_data = create_pdf(season, best_p, avoid_p, hex_skin)
            st.download_button(label="📥 Download PDF Report", data=pdf_data, file_name=f"Color_Report_{season}.pdf", mime="application/pdf")

        with col_left:
            # Main Draping View
            draped = apply_drape(processed_img, st.session_state.drape_color)
            st.image(cv2.cvtColor(draped, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Palette Tabs
            tab1, tab2 = st.tabs(["✨ Best Palette", "✕ Avoid Palette"])
            with tab1:
                t1_cols = st.columns(len(best_p))
                for i, c in enumerate(best_p):
                    t1_cols[i].markdown(f"<div class='color-block' style='background-color:{c};'></div>", unsafe_allow_html=True)
                    t1_cols[i].caption(c)
            with tab2:
                t2_cols = st.columns(len(avoid_p))
                for i, c in enumerate(avoid_p):
                    t2_cols[i].markdown(f"<div class='avoid-block' style='background-color:{c};'></div>", unsafe_allow_html=True)
                    t2_cols[i].caption(c)
    else:
        st.error("Face not found. Please use a clearer photo.")