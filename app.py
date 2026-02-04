import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
from color_utils import get_face_data, analyze_12_seasons, apply_drape, PALETTES, AVOID_PALETTES

st.set_page_config(page_title="Color Studio", layout="wide")

# Custom CSS for visual blocks
st.markdown("""
    <style>
    .color-swatch { border-radius: 5px; height: 60px; border: 1px solid #ddd; margin-bottom: 5px; }
    .stButton>button { width: 100%; height: 45px; border: none; color: transparent; }
    </style>
""", unsafe_allow_html=True)

def create_pdf(season, best_colors, avoid_colors, hex_skin):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "Color Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", '', 16)
    pdf.cell(0, 10, f"Detected Season: {season}", ln=True, align='C')
    pdf.cell(0, 10, f"Skin Tone Code: {hex_skin}", ln=True, align='C')
    
    pdf.ln(20)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Best Harmonious Palette:", ln=True)
    for color in best_colors:
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_fill_color(r, g, b)
        pdf.rect(pdf.get_x(), pdf.get_y(), 30, 15, 'F')
        pdf.set_x(pdf.get_x() + 35)
    
    pdf.ln(30)
    pdf.cell(0, 10, "Colors to Avoid:", ln=True)
    for color in avoid_colors:
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_fill_color(r, g, b)
        pdf.rect(pdf.get_x(), pdf.get_y(), 30, 15, 'F')
        pdf.set_x(pdf.get_x() + 35)
        
    return pdf.output() # This returns a bytearray

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
        best_p, avoid_p = PALETTES[season], AVOID_PALETTES[season]
        hex_skin = '#%02x%02x%02x' % tuple(avg_rgb)

        col_left, col_right = st.columns([2, 1])

        with col_right:
            st.header("Results")
            st.success(f"You are a **{season}**")
            st.markdown(f"**Skin Sample:** <div class='color-swatch' style='background-color:{hex_skin};'></div>", unsafe_allow_html=True)
            
            st.write("### Test Drapes")
            st.write("Click a color to see it against your face:")
            drape_grid = st.columns(4)
            for idx, color in enumerate(best_p + avoid_p):
                with drape_grid[idx % 4]:
                    # Create a button that looks like the color it selects
                    st.markdown(f"<div style='background-color:{color}; height:10px; border-radius:5px 5px 0 0;'></div>", unsafe_allow_html=True)
                    if st.button(" ", key=f"btn_{idx}"):
                        st.session_state.drape_color = color
            
            st.write("---")
            # FIXED: Wrapped in bytes() to solve the StreamlitAPIException
            pdf_data = bytes(create_pdf(season, best_p, avoid_p, hex_skin))
            st.download_button(label="📥 Download PDF Report", data=pdf_data, file_name=f"Report_{season}.pdf", mime="application/pdf")

        with col_left:
            draped = apply_drape(processed_img, st.session_state.drape_color)
            st.image(cv2.cvtColor(draped, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            tab1, tab2 = st.tabs(["✨ Best Palette", "✕ Avoid Palette"])
            with tab1:
                t1_cols = st.columns(len(best_p))
                for i, c in enumerate(best_p):
                    t1_cols[i].markdown(f"<div class='color-swatch' style='background-color:{c};'></div>", unsafe_allow_html=True)
                    t1_cols[i].caption(c)
            with tab2:
                t2_cols = st.columns(len(avoid_p))
                for i, c in enumerate(avoid_p):
                    t2_cols[i].markdown(f"<div class='color-swatch' style='background-color:{c}; filter:grayscale(30%);'></div>", unsafe_allow_html=True)
                    t2_cols[i].caption(c)
    else:
        st.error("Face not found. Please use a clearer photo.")