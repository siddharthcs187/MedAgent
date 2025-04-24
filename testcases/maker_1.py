import os
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import numpy as np

# Create directory
path = './patient_1/data/'
os.makedirs(path, exist_ok=True)

# 1. PNG note
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)
text = """Patient Name: Jane Doe
Age: 68 y/o   Sex: F
Hx: HTN x10y, T2DM x5y
CC: Fatigue, RUQ pain
Exam: BP 150/90 mmHg, HR 88 bpm
Plan: Order CBC, LFTs, CT chest, biopsy if indicated."""
draw.text((10, 10), text, fill='black')
img_path = path + 'patient_1_note.png'
img.save(img_path)

# 2. CT PDF
ct_path = path + 'patient_1_ct_report.pdf'
c = canvas.Canvas(ct_path, pagesize=letter)
c.drawString(30, 750, "Clinical Information: 68-year-old female with RUQ pain and elevated LFTs.")
c.drawString(30, 720, "Technique: Contrast-enhanced helical CT from lung apices to bases.")
c.showPage()
c.drawString(30, 750, "Findings: Liver: mild enlargement, homogeneous attenuation.")
c.drawString(30, 735, "Impression: Mild hepatomegaly without focal lesion.")
c.save()

# 3. Pathology PDF
path_path = path + 'patient_1_pathology.pdf'
c = canvas.Canvas(path_path, pagesize=letter)
c.drawString(30, 750, "Gross: Three cores, tan-brown, length 1.8 cm.")
c.showPage()
c.drawString(30, 750, "Microscopic: Mild portal inflammation, interface hepatitis.")
c.drawString(30, 735, "Diagnosis: Chronic hepatitis with steatosis (Grade 2, Stage 1).")
c.save()

# 4. Excel lab results
df = pd.DataFrame({
    "Test": ["WBC","RBC","Hemoglobin (Hb)","Hematocrit (Hct)","ALT (SGPT)","AST (SGOT)","Alkaline Phosphatase","Albumin","Total Bilirubin","Prothrombin Time (PT)"],
    "Value":[11.2,4.8,12.8,38.0,78,65,120,3.2,1.5,13.0],
    "Units":["x10³/µL","×10⁶/µL","g/dL","%","U/L","U/L","U/L","g/dL","mg/dL","seconds"],
    "Normal Range":["4.0–10.0","4.0–5.4","11.5–15.5","35–45","7–55","8–48","40–129","3.5–5.0","0.1–1.2","9.4–12.5"]
})
xlsx_path = path + 'patient_1_labs.xlsx'
df.to_excel(xlsx_path, index=False)

# 5. Dummy DICOM file
dcm_path = path + 'patient_1_chest.dcm'
with open(dcm_path, 'wb') as f:
    f.write(b'DICM-----FAKE DICOM FILE FOR TESTING-----')

# List files
os.listdir(path)
