import streamlit as st
try:
    import pytesseract
    import cv2
except ImportError as e:
    st.error(f"Required library missing: {e}. Please contact the app administrator.")
    st.stop()
from PIL import Image
import numpy as np
import re

# Page config
st.set_page_config(page_title="Reportslelo", layout="centered")

st.title("🧾 Reportslelo - Lab Report Analyzer")
st.markdown("##### Harish Choudhary Clinic | 📞 8209558359")

uploaded_file = st.file_uploader("कृपया रिपोर्ट इमेज अपलोड करें 📤", type=["jpg", "png", "jpeg"])

def preprocess_image(image_cv):
    """Preprocess image for better OCR accuracy."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def extract_details(text):
    name_match = re.search(r'(?:Name|Patient Name|नाम)[:\- ]+([A-Za-z\s\.]+)', text)
    age_match = re.search(r'(?:Age|उम्र)[:\- ]+(\d+)', text)
    phone_match = re.search(r'(?:Phone|Mobile|मोबाइल)[:\- ]+(\d{10})', text)

    name = name_match.group(1).strip() if name_match else "Mr/Ms"
    age = age_match.group(1).strip() if age_match else "N/A"
    phone = phone_match.group(1).strip() if phone_match else "N/A"

    return name, age, phone

def generate_summary(text):
    text_lower = text.lower()
    summary = []

    if "glucose" in text_lower or "sugar" in text_lower:
        summary.append("ग्लूकोज़ स्तर रिपोर्ट में पाया गया है। अगर यह सामान्य सीमा से ऊपर है, तो यह डायबिटीज का संकेत हो सकता है।")

    if "hemoglobin" in text_lower:
        summary.append("हीमोग्लोबिन स्तर की जाँच की गई है। यह शरीर में खून की गुणवत्ता का संकेत देता है।")

    if "cholesterol" in text_lower:
        summary.append("कोलेस्ट्रॉल की मात्रा रिपोर्ट में है। अधिक कोलेस्ट्रॉल दिल की बीमारियों का कारण बन सकता है।")

    if "creatinine" in text_lower:
        summary.append("क्रिएटिनिन किडनी की सेहत का संकेत देता है। इसका स्तर सामान्य होना ज़रूरी है।")

    if not summary:
        summary.append("रिपोर्ट सामान्य लग रही है। लेकिन कोई भी लक्षण हो तो डॉक्टर से सलाह लें।")

    return "\n".join(summary)

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="अपलोड की गई रिपोर्ट", use_column_width=True)
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image = preprocess_image(image_cv)
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng+hin')

        if not extracted_text.strip():
            st.warning("कोई टेक्स्ट नहीं निकाला गया। कृपया स्पष्ट इमेज अपलोड करें।")
            st.stop()

        st.subheader("📄 रिपोर्ट से निकाला गया टेक्स्ट:")
        st.text(extracted_text)

        name, age, phone = extract_details(extracted_text)
        report_summary = generate_summary(extracted_text)

        final_message = f"""👤 नाम: {name}
🎂 उम्र: {age} साल
📱 संपर्क: {phone}

📑 रिपोर्ट का सारांश:
{report_summary}

🏥 Harish Choudhary Clinic
📞 8209558359"""

        st.subheader("📲 मरीज को भेजे जाने वाला मैसेज:")
        st.text(final_message)
    except Exception as e:
        st.error(f"इमेज प्रोसेसिंग में त्रुटि: {e}")
