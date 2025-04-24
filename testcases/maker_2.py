import os
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import numpy as np
import datetime

# Create directory
path = './patient_2/data/'
os.makedirs(path, exist_ok=True)

# 1. PNG consult note
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)
note = """Patient Name: John Smith
Age: 55 y/o   Sex: M
Hx: Obesity BMI 32, HTN x5y, Pre-DM (A1c 6.2%)
CC: Fatigue, RUQ discomfort, LE edema
Exam: BP 145/95 mmHg, HR 80 bpm, BMI 32
Plan: Order extended labs, US liver, CT abd/pelvis."""
draw.text((10, 10), note, fill='black')
img.save(path + 'patient2_note.png')

# 2. PDF-1: Abdominal USG (NAFLD)
usg = canvas.Canvas(path + 'patient2_ultrasound.pdf', pagesize=letter)
usg.drawString(30, 750, "Clinical History: 55-year-old M with obesity, elevated LFTs, RUQ discomfort.")  # :contentReference[oaicite:1]{index=1}
usg.drawString(30, 730, "Technique: Standard abdominal ultrasound, right upper quadrant views.")  # :contentReference[oaicite:2]{index=2}
usg.showPage()
usg.drawString(30, 750, "Findings:")  
usg.drawString(50, 735, "- Liver: increased echogenicity, poor visualization of portal vein walls (Grade 2 steatosis).")  
usg.drawString(50, 720, "- Spleen, gallbladder: normal.")  
usg.drawString(30, 690, "Impression: Moderate hepatic steatosis (Grade 2 NAFLD). Recommend LFT correlation & consider biopsy.")  
usg.save()

# 3. PDF-2: CT Abdomen/Pelvis with contrast
ct = canvas.Canvas(path + 'patient2_ct_abdomen.pdf', pagesize=letter)
ct.drawString(30, 750, "Indication: RUQ discomfort, elevated LFTs, obesity.")  
ct.drawString(30, 730, "Technique: Contrast-enhanced CT abdomen/pelvis, 5 mm axial slices.")  # :contentReference[oaicite:3]{index=3}
ct.showPage()
ct.drawString(30, 750, "Findings:")  
ct.drawString(50, 735, "- Liver: hepatomegaly, attenuation 45 HU (low density vs spleen 55 HU).")  # :contentReference[oaicite:4]{index=4}
ct.drawString(50, 720, "- Kidneys: cortical thinning, small cyst right kidney.")  
ct.drawString(50, 705, "- Adrenals, pancreas: unremarkable.")  
ct.drawString(30, 675, "Impression: Consistent with moderate hepatic steatosis; early cortical changes suggest CKD stage 2.")  
ct.save()

# 4. Excel extended labs
df = pd.DataFrame({
    "Test": [
        "WBC","RBC","Hb","Hct","Platelets",
        "ALT","AST","ALP","Albumin","Total Bilirubin",
        "Creatinine","BUN","eGFR","Glucose (Fasting)","HbA1c",
        "Total Cholesterol","LDL","HDL","Triglycerides"
    ],
    "Value": [
        7.8, 5.0, 15.2, 45.0, 250,
        85, 70, 110, 3.8, 1.2,
        1.3, 18, 75, 110, 6.2,
        220, 140, 40, 180
    ],
    "Units": [
        "×10³/µL","×10⁶/µL","g/dL","%","×10³/µL",
        "U/L","U/L","U/L","g/dL","mg/dL",
        "mg/dL","mg/dL","mL/min/1.73m²","mg/dL","%",
        "mg/dL","mg/dL","mg/dL","mg/dL"
    ],
    "Normal Range": [
        "4.0–10.0","4.0–5.4","13.5–17.5","41–53","150–450",
        "7–55","8–48","40–129","3.5–5.0","0.1–1.2",
        "0.5–1.1","6–21",">90","70–100","<5.7",
        "<200","<100",">60","<150"
    ]
})
df.to_excel(path + 'patient2_labs.xlsx', index=False)

# 5. Dummy DICOM CT slice
with open(path + 'patient2_ct_slice.dcm', 'wb') as f:
    f.write(b'DICM_FAKE_CT_SLICE_FOR_TESTING')

print("Generated:", os.listdir(path))
