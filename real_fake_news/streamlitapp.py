import streamlit as st
from newsdetection import predict_and_explain

# --- Page setup ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Initialize session state for input_text ---
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- Sidebar with sample articles ---
with st.sidebar:
    st.markdown("### Try Sample Articles")
    if st.button("Real News Example"):
        st.session_state.input_text = "The government announced a new education policy aimed at improving literacy rates."
    if st.button("Fake News Example"):
        st.session_state.input_text = "Aliens landed in Texas and offered a cure for cancer using crystals."

# --- App title and instructions ---
st.title(" Fake News Detector")
st.markdown("Enter a news article or snippet below to check if it's real or fake.")

# --- Main input area (uses session state) ---
input_text = st.text_area(
    "Enter news article text here:",
    value=st.session_state.input_text,
    height=200
)

# --- Prediction ---
if st.button("Check"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        prediction, explanation = predict_and_explain(input_text)
        if prediction == "Real News":
            st.success(" This article appears to be **Real News**.")
        else:
            st.error(" This article appears to be **Fake News**.")
        st.markdown(f"**Top contributing words:** `{explanation}`")
