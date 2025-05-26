import streamlit as st
from PIL import Image
import pytesseract
import re

# Page settings
st.set_page_config(page_title="Reportslelo", layout="centered")

st.title("🧾 Reportslelo - Lab Report Summary Generator")
st.subheader("Harish Choudhary Clinic 📞 8209558359")

# Upload image
uploaded_file = st.file_uploader("अपनी लैब रिपोर्ट अपलोड करें (फोटो या स्कैन की गई)", type=["jpg", "jpeg", "png"])

def generate_summary(text):
    summary_lines = []

    # Patient name detect
    name_match = re.search(r'(Name|नाम)[:\s\-]+([A-Za-z\s]+)', text)
    if name_match:
        name = name_match.group(2).strip()
        summary_lines.append(f"प्रिय {name}, आपकी लैब रिपोर्ट निम्नलिखित है:\n")
    else:
        summary_lines.append("प्रिय मरीज, आपकी लैब रिपोर्ट निम्नलिखित है:\n")

    explained_tests = 0

    # ESR detection
    esr_match = re.search(r'ESR[:\s\-]+([\d.]+)', text, re.IGNORECASE)
    if esr_match:
        value = float(esr_match.group(1))
        if value <= 20:
            summary_lines.append(f"👉 ESR: {value} mm/hr (सामान्य)\nESR की सामान्य सीमा 0–20 mm/hr होती है। घबराने की कोई बात नहीं है।")
        else:
            summary_lines.append(f"👉 ESR: {value} mm/hr (थोड़ा बढ़ा हुआ)\nESR सामान्य से अधिक है, यह सूजन या संक्रमण का संकेत हो सकता है। डॉक्टर से सलाह लें।")
        explained_tests += 1

    # Hemoglobin detection
    hb_match = re.search(r'Hemoglobin[:\s\-]+([\d.]+)', text, re.IGNORECASE)
    if hb_match:
        value = float(hb_match.group(1))
        if value >= 13:
            summary_lines.append(f"\n👉 हीमोग्लोबिन: {value} g/dL (सामान्य)\nहीमोग्लोबिन की सामान्य सीमा पुरुषों के लिए 13–17 g/dL होती है।")
        else:
            summary_lines.append(f"\n👉 हीमोग्लोबिन: {value} g/dL (कम)\nहीमोग्लोबिन कम है। यह एनीमिया या कमजोरी का कारण हो सकता है। डॉक्टर से मिलें।")
        explained_tests += 1

    # Sugar detection
    sugar_match = re.search(r'Sugar[:\s\-]+([\d.]+)', text, re.IGNORECASE)
    if sugar_match:
        value = float(sugar_match.group(1))
        if value <= 140:
            summary_lines.append(f"\n👉 ब्लड शुगर: {value} mg/dL (सामान्य)\nशुगर सामान्य सीमा में है।")
        else:
            summary_lines.append(f"\n👉 ब्लड शुगर: {value} mg/dL (ऊंचा)\nशुगर बढ़ा हुआ है, यह डायबिटीज का संकेत हो सकता है। डॉक्टर से मिलें।")
        explained_tests += 1

    # If nothing found
    if explained_tests == 0:
        summary_lines.append("आपकी रिपोर्ट सामान्य प्रतीत होती है। कोई चिंता की बात नहीं है। यदि कुछ असामान्य लगे तो डॉक्टर से सलाह लें।")

    summary_lines.append("\n\n- Harish Choudhary Clinic\n📞 8209558359")

    return "\n".join(summary_lines)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='अपलोड की गई रिपोर्ट', use_container_width=True)

    with st.spinner("रिपोर्ट पढ़ी जा रही है..."):
        extracted_text = pytesseract.image_to_string(image, lang='eng')
        st.subheader("📄 Extracted Report Text:")
        st.text(extracted_text)

        st.subheader("📩 Patient को भेजने योग्य सरल हिंदी सारांश:")
        summary = generate_summary(extracted_text)
        st.text_area("SMS Preview", summary, height=300)
