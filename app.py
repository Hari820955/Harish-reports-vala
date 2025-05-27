import streamlit as st
import pytesseract
from PIL import Image
import re
import os
from io import BytesIO

# -----------------------------
# OCR Setup (Tesseract path not needed on Streamlit Cloud)
# -----------------------------
st.set_page_config(page_title="Reportslelo - Harish Choudhary Clinic", layout="centered")

st.title("🧾 Reportslelo - Lab Report Summary Generator")
st.caption("by Harish Choudhary Clinic | Contact: 8209558359")

uploaded_file = st.file_uploader("🖼️ Lab Report Photo Upload karo (Camera ya Gallery se)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Report", use_container_width=True)

    with st.spinner("🔍 Report padhi ja rahi hai, kripya ruk jao..."):
        # OCR: extract text from image
        extracted_text = pytesseract.image_to_string(image)

    st.subheader("📃 Extracted Report Text")
    st.text_area("Yeh report se mila text:", value=extracted_text, height=200)

    # ----------------------------------
    # Info Extraction: Patient name, age, contact (if found)
    # ----------------------------------
    name_match = re.search(r"Name[:\-\s]*([A-Za-z ]+)", extracted_text)
    age_match = re.search(r"Age[:\-\s]*(\d+)", extracted_text)
    mobile_match = re.search(r"[\+]?91[\- ]?[6-9]\d{9}|[6-9]\d{9}", extracted_text)

    name = name_match.group(1).strip() if name_match else "Patient"
    age = age_match.group(1).strip() if age_match else "N/A"
    mobile = mobile_match.group(0).strip() if mobile_match else "N/A"

    # ----------------------------------
    # Smart Summary Logic (based on keywords in text)
    # ----------------------------------
    summary_lines = []

    # Sample explanations
    if "ESR" in extracted_text.upper():
        esr_match = re.search(r"ESR[:\s]+(\d+\.?\d*)", extracted_text, re.IGNORECASE)
        if esr_match:
            esr = float(esr_match.group(1))
            if esr <= 20:
                summary_lines.append(f"Aapka ESR {esr} mm/hr hai, jo samanya range mein hai.")
            else:
                summary_lines.append(f"Aapka ESR {esr} mm/hr hai, jo thoda adhik hai. Doctor se salah lena uchit hoga.")

    if "HEMOGLOBIN" in extracted_text.upper():
        hb_match = re.search(r"Hemoglobin[:\s]+(\d+\.?\d*)", extracted_text, re.IGNORECASE)
        if hb_match:
            hb = float(hb_match.group(1))
            if hb >= 12:
                summary_lines.append(f"Aapka Hemoglobin {hb} g/dL hai, jo achha hai.")
            else:
                summary_lines.append(f"Aapka Hemoglobin {hb} g/dL hai, jo kam hai. Aapko iron rich diet leni chahiye.")

    if "WBC" in extracted_text.upper():
        wbc_match = re.search(r"WBC[:\s]+(\d+,?\d+)", extracted_text, re.IGNORECASE)
        if wbc_match:
            wbc = wbc_match.group(1).replace(",", "")
            wbc = int(wbc)
            if 4000 <= wbc <= 11000:
                summary_lines.append(f"WBC Count {wbc}/µL hai, jo normal hai.")
            else:
                summary_lines.append(f"WBC Count {wbc}/µL hai, jo abnormal ho sakta hai. Doctor se salah lein.")

    if not summary_lines:
        summary_lines.append("Report samanya lag rahi hai. Kisi bhi shak hone par doctor se salah lein.")

    # ----------------------------------
    # Final Message to Send
    # ----------------------------------
    st.subheader("📩 Patient Ko Bhejne Wala Message")
    final_msg = f"""
    👤 Naam: {name}
    🎂 Umar: {age} saal
    📱 Contact: {mobile}

    📑 Report ka Saar:
    {'\n'.join(summary_lines)}

    🏥 Harish Choudhary Clinic
    📞 8209558359
    """

    st.text_area("Final SMS/Message to Patient:", value=final_msg.strip(), height=250)

    st.success("✅ Yeh message aap SMS ya WhatsApp se bhej sakte ho patient ko.")

