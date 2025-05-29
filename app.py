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
    # Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    # Try multiple preprocessing methods
    # Method 1: Adaptive thresholding with increased block size
    thresh1 = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    # Method 2: Otsu thresholding with blur
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Method 3: Simple binary thresholding
    _, thresh3 = cv2.threshold(contrast, 150, 255, cv2.THRESH_BINARY)
    # Denoise all
    denoised1 = cv2.fastNlMeansDenoising(thresh1)
    denoised2 = cv2.fastNlMeansDenoising(thresh2)
    denoised3 = cv2.fastNlMeansDenoising(thresh3)
    # Resize (scale up 2x for better OCR)
    resized1 = cv2.resize(denoised1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    resized2 = cv2.resize(denoised2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    resized3 = cv2.resize(denoised3, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return resized1, resized2, resized3

def extract_details(text):
    """Extract name, age, and phone number with robust regex."""
    # Name: Match variations or standalone names
    name_match = re.search(r'(?:Name|Patient Name|नाम|[A-Za-z]+\.)[:\- ]*([A-Za-z\s\.]+)|([A-Za-z\s\.]+)(?=\s*(?:Age|उम्र))|\b([A-Za-z\s\.]{3,20})\b', text, re.IGNORECASE)
    # Age: Match "Age <: 21 Years", "Age: 21", "उम्र: 21", or standalone numbers
    age_match = re.search(r'(?:Age|उम्र)[:\<\-\s]+(\d+)(?:\s*Years)?|\b(\d{1,3})\s*(?:Years|yrs)', text, re.IGNORECASE)
    # Phone: Match 10-digit numbers
    phone_match = re.search(r'(?:Phone|Mobile|मोबाइल)[:\- ]*(\d{10})|\b(\d{10})\b', text, re.IGNORECASE)

    name = (name_match.group(1) or name_match.group(2) or name_match.group(3)).strip() if name_match else "Mr/Ms"
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

    # Reference ranges (simplified)
    ranges = {
        "glucose": (70, 140), "hba1c": (4, 5.6), "fbs": (70, 100), "ppbs": (70, 140), "rbs": (70, 140),
        "cholesterol": (0, 200), "creatinine": (0.6, 1.2), "urea": (10, 50), "uric acid": (3.5, 7.2),
        "vitamin d": (20, 50), "vitamin b12": (200, 900), "tsh": (0.4, 4.0), "t3": (80, 200), "t4": (5, 12),
        "esr": (0, 20), "crp": (0, 10), "ldh": (140, 280), "ferritin": (30, 400), "tibc": (250, 450),
        "hemoglobin": (12, 16)
    }

    # Extract test names and values with error handling
    detected_tests = []
    try:
        test_matches = re.findall(r'(\w+(?:\s+\w+)*)\s*[:\-=]\s*([\d\.]+)\s*(\w+)?', text, re.IGNORECASE)
        if not isinstance(test_matches, list):
            raise ValueError("Regex did not return a list of matches.")
        for test_name, value, unit in test_matches:
            try:
                value = float(value)
                test_key = test_name.lower().strip()
                for test in tests:
                    if test in test_key:
                        for range_key, (low, high) in ranges.items():
                            if range_key in test_key:
                                status = "सामान्य" if low <= value <= high else "असामान्य"
                                detected_tests.append(f"{test_name} का स्तर {value} {unit or ''} है, जो {status} है।")
                                break
                        else:
                            detected_tests.append(f"{test_name} का स्तर {value} {unit or ''} है। डॉक्टर से इसकी व्याख्या करवाएं।")
                        break
            except (ValueError, TypeError):
                continue
    except Exception as e:
        st.warning(f"टेस्ट डिटेक्शन में त्रुटि: {e}. OCR आउटपुट की जाँच करें।")
        detected_tests = []

    # Add detected tests to summary
    summary.extend(detected_tests[:2])

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
    if "vitamin d" in text_lower or "vitamin b12" in text_lower:
        summary.append("विटामिन डी या बी12 की कमी हड्डियों और तंत्रिका स्वास्थ्य को प्रभावित कर सकती है।")

    # Fallback if no tests are detected or OCR fails
    if not summary:
        summary.extend([
            "रिपोर्ट से टेक्स्ट ठीक से नहीं निकाला जा सका।",
            "कृपया उच्च-रिज़ॉल्यूशन (कम से कम 300 DPI) और स्पष्ट इमेज अपलोड करें।",
            "रिपोर्ट की पूरी व्याख्या के लिए डॉक्टर से परामर्श लें।",
            "नियमित स्वास्थ्य जांच और संतुलित आहार बनाए रखें।",
            "किसी भी लक्षण जैसे थकान, पेट दर्द या बुखार के लिए तुरंत डॉक्टर से संपर्क करें।",
            "स्वस्थ जीवनशैली अपनाएं और क्लिनिक से संपर्क करें।"
        ])

    # Ensure 5-6 lines
    return "\n".join(summary[:6])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="अपलोड की गई रिपोर्ट", use_container_width=True)
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Get multiple preprocessed images
        processed_image1, processed_image2, processed_image3 = preprocess_image(image_cv)
        
        # Try multiple OCR configurations
        extracted_text = pytesseract.image_to_string(processed_image1, lang='eng')
        if len(extracted_text.strip()) < 30:  # Increased threshold for meaningful text
            extracted_text = pytesseract.image_to_string(processed_image1, lang='eng+hin')
        if len(extracted_text.strip()) < 30:
            extracted_text = pytesseract.image_to_string(processed_image2, lang='eng')
        if len(extracted_text.strip()) < 30:
            extracted_text = pytesseract.image_to_string(processed_image2, lang='eng+hin')
        if len(extracted_text.strip()) < 30:
            extracted_text = pytesseract.image_to_string(processed_image3, lang='eng')
        if len(extracted_text.strip()) < 30:
            extracted_text = pytesseract.image_to_string(processed_image3, lang='eng+hin')

        if len(extracted_text.strip()) < 30:
            # Fallback summary if OCR fails completely
            st.warning("कोई टेक्स्ट ठीक से नहीं निकाला गया। कृपया उच्च-रिज़ॉल्यूशन (कम से कम 300 DPI), स्पष्ट इमेज अपलोड करें।")
            name, age, phone = "Mr/Ms", "N/A", "N/A"
            report_summary = "\n".join([
                "रिपोर्ट से टेक्स्ट ठीक से नहीं निकाला जा सका।",
                "कृपया उच्च-रिज़ॉल्यूशन (कम से कम 300 DPI) और स्पष्ट इमेज अपलोड करें।",
                "रिपोर्ट की पूरी व्याख्या के लिए डॉक्टर से परामर्श लें।",
                "नियमित स्वास्थ्य जांच और संतुलित आहार बनाए रखें।",
                "किसी भी लक्षण जैसे थकान, पेट दर्द या बुखार के लिए तुरंत डॉक्टर से संपर्क करें।",
                "स्वस्थ जीवनशैली अपनाएं और क्लिनिक से संपर्क करें।"
            ])
        else:
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
