from llama_cpp import Llama


llm = Llama.from_pretrained(
    repo_id="aaditya/OpenBioLLM-Llama3-8B-GGUF",
    filename="openbiollm-llama3-8b.Q4_K_M.gguf",
    n_gpu_layers=30,
    n_ctx=8192,
    verbose=False
)

summary_response = """
content='### Patient Overview\n\n- **Demographics:** 70-year-old Caucasian male.\n- **Complaint:**  Hemorrhage/hematoma, intracranial (admission diagnosis).  Presented with altered mental status, obtundation, and hemorrhagic stroke (internal capsule) likely secondary to hypertension.  Also presented with chronic renal insufficiency.\n\n### Clinical Findings\n\n**Vitals (from Physical Exam, 19 minutes post-unit admission):**\n\n* Heart Rate: 78 bpm\n* Blood Pressure: 163/115 mmHg (highest recorded)\n* Respiratory Rate: 45 breaths/minute\n* Oxygen Saturation: 95%\n* Weight: 120.2 kg\n* Glasgow Coma Scale (GCS): Scored; Motor 6, Eyes 4, Verbal 4 (meaning of "scored" unclear).\n* Intake/Output: Intake 460 ml, Output 1 ml, Net +459 ml.\n\n**Labs:**  (Note:  Time offsets are provided but without a specific reference time, making precise temporal interpretation difficult. Values represent a range of results over the patient\'s stay.)\n\n* **Hematology:**  Hgb (14.8-16.6 g/dL), Hct (44.3-50.6%), WBC (10.6-13.3 K/mcL), RBC (4.93-5.61 M/mcL), MCV (89.9-90.5 fL), MCH (29.4-29.9 pg), MCHC (32.6-33.3 g/dL), Platelets (194-207 K/mcL), RDW (16.3-16.5%), Differential counts available (lymphocytes, monocytes, polymorphonuclear leukocytes, eosinophils, basophils) with values within normal ranges (except for slight variation in lymphocytes and monocytes).\n* **Chemistry:** Glucose (121-200 mg/dL; highly variable), BUN (17-21 mg/dL), Creatinine (1.12-1.6 mg/dL), Sodium (137-140 mmol/L), Potassium (3.6-4 mmol/L), Chloride (99-103 mmol/L), Calcium (8.9-9 mg/dL), Ionized Calcium (4.6 mg/dL), Anion Gap (8-13), Lactate (1.2-2 mmol/L), Fibrinogen (436 mg/dL).\n* **Coagulation:** PT (22.3-25.8 sec), PT-INR (2.0-2.3).\n* **Urinalysis:** Urinary Specific Gravity 1.01.\n\n\n### Summary & Recommendations\n\nThis 70-year-old male was admitted to the Cardiac ICU following a hemorrhagic stroke (internal capsule) likely related to his pre-existing hypertension. He also presented with altered mental status (obtundation), and chronic renal insufficiency.  His initial GCS score suggests moderate to severe neurological impairment.  The highly variable glucose levels warrant further investigation.  Elevated creatinine and BUN levels indicate renal dysfunction, which should be closely monitored.  The elevated PT-INR suggests a potential coagulopathy.  The provided data lacks sufficient detail to fully assess the temporal relationships of lab results and their correlation with the patient\'s clinical course.  A complete review of the patient\'s chart, including missing vital sign trends and a clear interpretation of the GCS score, is necessary for a more comprehensive assessment.  Further investigation into the cause of the glucose variability is crucial.  The patient\'s response to treatment (Nicardipine for hypertension) should be carefully evaluated.  Given the severity of the hemorrhagic stroke and the presence of renal insufficiency, ongoing neurological and renal monitoring are critical."""

prompt = f"""<s>[INST] You are an experienced clinical decision support assistant developed by Saama AI Labs. Based on the following patient summary, identify the most important actionable clinical insights. These should include recommendations for diagnosis refinement, immediate interventions, further investigations, and potential referrals. Be specific, evidence-based, and avoid vague statements.

Patient Summary:
\"\"\"
{summary_response}
\"\"\"
Provide a bullet-point list of actionable clinical insights for this patient.[/INST]
"""
output = llm(prompt, max_tokens=512)

print(output['choices'][0]['text'].strip())
