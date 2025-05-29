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
    """Preprocess image for robust OCR, especially low-quality images."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    # Resize (scale up 2x for better OCR)
    resized = cv2.resize(denoised, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return resized

def extract_details(text):
    """Extract name, age, and phone number with robust regex."""
    # Name: Match variations like "Mr.HARISH", "Name: HARISH", "नाम: हरिश", or before Age
    name_match = re.search(r'(?:Name|Patient Name|नाम|[A-Za-z]+\.)[:\- ]*([A-Za-z\s\.]+)|([A-Za-z\s\.]+)(?=\s*(?:Age|उम्र))', text, re.IGNORECASE)
    # Age: Match "Age <: 21 Years", "Age: 21", "उम्र: 21", or standalone near "Years"
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

    # List of tests to detect
    tests = [
        "cbc", "complete blood count", "esr", "erythrocyte sedimentation rate", "crp", "c-reactive protein",
        "hba1c", "fasting blood sugar", "fbs", "postprandial blood sugar", "ppbs", "random blood sugar", "rbs",
        "lipid profile", "lft", "liver function test", "kft", "kidney function test", "serum creatinine",
        "blood urea", "uric acid", "vitamin d", "vitamin b12", "tsh", "t3", "t4", "testosterone", "estrogen",
        "fsh", "lh", "prolactin", "insulin fasting", "ana", "anti-nuclear antibody", "anca", "ldh",
        "lactate dehydrogenase", "iron studies", "serum ferritin", "tibc", "total iron binding capacity",
        "urine routine", "urine microscopy", "urine culture", "microalbuminuria", "dengue ns1 antigen",
        "dengue igg", "dengue igm", "malaria parasite", "mp test", "widal test", "typhoid", "hbsag",
        "hepatitis b surface antigen", "anti-hcv", "hepatitis c", "hiv i", "hiv ii", "vdrl", "syphilis",
        "rpr", "rapid plasma reagin", "covid-19 rt-pcr", "covid-19 antibody", "d-dimer", "pt/inr",
        "prothrombin time", "aptt", "activated partial thromboplastin time", "blood grouping", "rh typing",
        "ecg", "chest x-ray", "x-ray", "ultrasound abdomen", "mri brain", "ct scan chest", "pap smear",
        "semen analysis"
    ]
    
    # Reference ranges (simplified, customize as needed)
    ranges = {
        "glucose": (70, 140), "hba1c": (4, 5.6), "fbs": (70, 100), "ppbs": (70, 140), "rbs": (70, 140),
        "cholesterol": (0, 200), "creatinine": (0.6, 1.2), "urea": (10, 50), "uric acid": (3.5, 7.2),
        "vitamin d": (20, 50), "vitamin b12": (200, 900), "tsh": (0.4, 4.0), "t3": (80, 200), "t4": (5, 12),
        "esr": (0, 20), "crp": (0, 10), "ldh": (140, 280), "ferritin": (30, 400), "tibc": (250, 450)
    }

    # Extract test names and values
    test_matches = re.findall(r'(\w+(?:\s+\w+)*)\s*[:\-=]\s*([\d\.]+)\s*(\w+)?', text, re.IGNORECASE)
    detected_tests = []

    for test_name, value, unit in test_matches:
        try:
            value = float(value)
            test_key = test_name.lower().strip()
            for test in tests:
                if test in test_key:
                    # Check if test has a known range
                    for range_key, (low, high) in ranges.items():
                        if range_key in test_key:
                            status = "सामान्य" if low <= value <= high else "असामान्य"
                            detected_tests.append(f"{test_name} का स्तर {value} {unit or ''} है, जो {status} है।")
                            break
                    else:
                        detected_tests.append(f"{test_name} का स्तर {value} {unit or ''} है। डॉक्टर से इसकी व्याख्या करवाएं।")
                    break
        except ValueError:
            continue

    # Add detected tests to summary
    summary.extend(detected_tests[:3])  # Limit to 3 to avoid overcrowding

    # Add general insights based on detected tests
    if any("cbc" in text_lower or "complete blood count" in text_lower):
        summary.append("सीबीसी टेस्ट रक्त कोशिकाओं की स्थिति दर्शाता है। असामान्यता एनीमिया या संक्रमण का संकेत हो सकती है।")
    if "esr" in text_lower or "crp" in text_lower:
        summary.append("ईएसआर या सीआरपी में वृद्धि सूजन या संक्रमण का संकेत हो सकती है।")
    if "lipid profile" in text_lower or "cholesterol" in text_lower:
        summary.append("लिपिड प्रोफाइल हृदय स्वास्थ्य के लिए महत्वपूर्ण है। उच्च स्तर जोखिम बढ़ा सकता है।")
    if "lft" in text_lower or "liver function" in text_lower:
        summary.append("लिवर फंक्शन टेस्ट यकृत स्वास्थ्य की जाँच करता है। असामान्यता पर ध्यान दें।")
    if "kft" in text_lower or "kidney function" in text_lower:
        summary.append("किडनी फंक्शन टेस्ट गुर्दे की सेहत दर्शाता है। डॉक्टर से परामर्श लें।")

    # General advice
    if not summary:
        summary.append("रिपोर्ट में कोई विशिष्ट असामान्यता नहीं दिख रही है।")
    summary.append("रिपोर्ट के परिणामों की पूरी व्याख्या के लिए डॉक्टर से परामर्श लें।")
    summary.append("नियमित स्वास्थ्य जांच और संतुलित आहार बनाए रखें।")

    # Ensure 5-6 lines
    if len(summary) < 5:
        summary.append("किसी भी लक्षण जैसे थकान, पेट दर्द या बुखार के लिए तुरंत डॉक्टर से संपर्क करें।")
    if len(summary) < 6:
        summary.append("स्वस्थ जीवनशैली अपनाएं और क्लिनिक से संपर्क करें।")

    return "\n".join(summary[:6])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="अपलोड की गई रिपोर्ट", use_container_width=True)
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image = preprocess_image(image_cv)
        
        # Try multiple OCR configurations
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng')
        if not extracted_text.strip():
            extracted_text = pytesseract.image_to_string(processed_image, lang='eng+hin')
        if not extracted_text.strip():
            # Try without resizing for some cases
            processed_image = cv2.adaptiveThreshold(
                cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY), 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            extracted_text = pytesseract.image_to_string(processed_image, lang='eng')

        if not extracted_text.strip():
            st.warning("कोई टेक्स्ट नहीं निकाला गया। कृपया उच्च-रिज़ॉल्यूशन, स्पष्ट इमेज अपलोड करें।")
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
