import streamlit as st
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
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    # Resize for better OCR (optional, scale up 1.5x)
    resized = cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return resized

def extract_details(text):
    """Extract name, age, and phone number with robust regex."""
    # Name: Match variations like "Mr.HARISH", "Name: HARISH", "नाम: हरिश", or standalone names
    name_match = re.search(r'(?:Name|Patient Name|नाम|[A-Za-z]+\.)[:\- ]*([A-Za-z\s\.]+)|([A-Za-z\s\.]+)(?=\s*(?:Age|उम्र))', text, re.IGNORECASE)
    # Age: Match "Age <: 21 Years", "Age: 21", "उम्र: 21", or standalone numbers near "Age"
    age_match = re.search(r'(?:Age|उम्र)[:\<\-\s]+(\d+)(?:\s*Years)?|\b(\d{1,3})\s*(?:Years|yrs)', text, re.IGNORECASE)
    # Phone: Match 10-digit numbers after "Phone", "Mobile", or standalone
    phone_match = re.search(r'(?:Phone|Mobile|मोबाइल)[:\- ]*(\d{10})|\b(\d{10})\b', text, re.IGNORECASE)

    name = (name_match.group(1) or name_match.group(2)).strip() if name_match else "Mr/Ms"
    age = (age_match.group(1) or age_match.group(2)).strip() if age_match else "N/A"
    phone = (phone_match.group(1) or phone_match.group(2)).strip() if phone_match else "N/A"

    # Debug extraction
    st.write("**Debug: Extracted Details**")
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    st.write(f"Phone: {phone}")

    return name, age, phone

def generate_summary(text):
    """Generate a 5-6 line AI-driven summary in Hindi for any lab report."""
    summary = []
    text_lower = text.lower()

    # Extract test names and values using regex
    test_matches = re.findall(r'(\w+(?:\s+\w+)*)\s*[:\-=]\s*([\d\.]+)\s*(\w+)?', text, re.IGNORECASE)
    abnormal_tests = []
    
    # Analyze tests and values
    for test_name, value, unit in test_matches:
        try:
            value = float(value)
            # Placeholder ranges (customize based on common tests)
            ranges = {
                "glucose": (70, 140),
                "hemoglobin": (12, 16),
                "cholesterol": (0, 200),
                "creatinine": (0.6, 1.2),
                "ttg": (0.01, 20.0)
            }
            test_key = test_name.lower().strip()
            for key, (low, high) in ranges.items():
                if key in test_key:
                    status = "सामान्य" if low <= value <= high else "असामान्य"
                    abnormal_tests.append(f"{test_name} का स्तर {value} {unit or ''} है, जो {status} है।")
                    break
            else:
                summary.append(f"{test_name} का स्तर {value} {unit or ''} है। डॉक्टर से इसकी व्याख्या करवाएं।")
        except ValueError:
            continue

    # Add specific insights
    if "ttg" in text_lower or "celiac" in text_lower:
        summary.append("रिपोर्ट में टीटीजी टेस्ट है, जो सेलियक रोग से संबंधित हो सकता है। ग्लूटेन-मुक्त आहार पर विचार करें।")
    if "glucose" in text_lower or "sugar" in text_lower:
        summary.append("ग्लूकोज़ स्तर की जाँच की गई है। असामान्य स्तर डायबिटीज का संकेत हो सकता है।")
    if abnormal_tests:
        summary.extend(abnormal_tests[:2])  # Limit to 2 to avoid overcrowding

    # General advice
    summary.append("रिपोर्ट के परिणामों की पूरी व्याख्या के लिए डॉक्टर से परामर्श लें।")
    summary.append("नियमित स्वास्थ्य जांच और संतुलित आहार बनाए रखें।")

    # Ensure 5-6 lines
    if len(summary) < 5:
        summary.append("किसी भी लक्षण जैसे थकान या पेट दर्द के लिए तुरंत डॉक्टर से संपर्क करें।")
    if len(summary) < 6:
        summary.append("स्वस्थ जीवनशैली अपनाएं और क्लिनिक से संपर्क करें।")

    return "\n".join(summary[:6])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="अपलोड की गई रिपोर्ट", use_container_width=True)
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image = preprocess_image(image_cv)
        
        # Try English first, fallback to English+Hindi
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng')
        if not extracted_text.strip():
            extracted_text = pytesseract.image_to_string(processed_image, lang='eng+hin')
        
        if not extracted_text.strip():
            st.warning("कोई टेक्स्ट नहीं निकाला गया। कृपया स्पष्ट, उच्च-रिज़ॉल्यूशन इमेज अपलोड करें।")
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
