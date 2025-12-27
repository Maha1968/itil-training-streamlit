import streamlit as st

st.set_page_config(layout="wide")

st.title("Nexware Technologies Private Limited – ITIL Training")

# Google Drive PDF (must be public)
PDF_URL = "https://drive.google.com/uc?export=download&id=1reuk-JLnou4D3rgDbhIA7fhDhqw8RZm9"

# Initialize page number
if "page" not in st.session_state:
    st.session_state.page = 1

# Navigation buttons
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("⬅ Previous"):
        if st.session_state.page > 1:
            st.session_state.page -= 1

with col3:
    if st.button("Next ➡"):
        st.session_state.page += 1

st.markdown(f"### Slide {st.session_state.page}")

# PDF.js Viewer
pdf_viewer = f"""
<iframe 
    src="https://mozilla.github.io/pdf.js/web/viewer.html?file={PDF_URL}#page={st.session_state.page}"
    width="100%"
    height="800px"
    style="border:none;">
</iframe>
"""

st.components.v1.html(pdf_viewer, height=820)

st.success("Training content loaded successfully.")
