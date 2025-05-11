from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import pandas as pd
from dotenv import load_dotenv
import os
import time

load_dotenv()

system_prompt = """
You are an expert medical-report evaluator.  You will receive two patient-history summaries (“Report1” and “Report2”).  
Your task is to score each summary on these three dimensions (0-10 scale) and compute an average.

CRITERIA:

1. Factual Accuracy  
   • Definition: Are all statements (diagnoses, lab values, dates, treatments) correct and traceable?  
   • Deduct heavily for any invented or reversed findings.  

2. Clinical Relevance  
   • Definition: Does the summary focus on elements that affect patient management (key symptoms, critical labs, major comorbidities, follow-up plans)?  
   • Down-score if trivial background or non-impactful minutiae dominate.  

3. Consistency  
   • Definition: Is the narrative internally coherent and medically plausible?  
   • Identify any self-contradictions (e.g. “stable BP” vs. “uncontrolled hypertension”).  

RUBRIC (map your judgment to an integer 0-10; after scoring each, compute the average as a float):

Scale interpretation:  
  • 9-10 = Excellent (no or trivial issues)  
  • 7-8 = Good (minor issues; clinically reliable)  
  • 5-6 = Adequate (some gaps; use with caution)  
  • 3-4 = Poor (major omissions/inaccuracies; not reliable)  
  • 0-2 = Unacceptable (critical errors; misleading)

Extreme scores require justification, do not arbitrarily award any score. 

For each report, return JSON exactly as:
{
  "report1": {
    "factual_accuracy": <int>,
    "clinical_relevance": <int>,
    "consistency": <int>,
    "average": <float>
  },
  "report2": {
    "factual_accuracy": <int>,
    "clinical_relevance": <int>,
    "consistency": <int>,
    "average": <float>
  }
}
"""

def compare_reports_with_gemini(report1: str, report2: str, model_name: str = "gemini-1.5-flash") -> dict:
    """
    Sends two medical-summary reports to Gemini and gets back scores (0-10) on:
      1. Factual Accuracy
      2. Clinical Relevance
      3. Consistency

    Returns a dict:
      {
        "report1": {"factual_accuracy": ...,
                    "clinical_relevance": ..., "consistency": ..., "average": ...},
        "report2": { ... same keys ... }
      }
    """

    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Report1:
\"\"\"{report1}\"\"\"

Report2:
\"\"\"{report2}\"\"\"

Please score both reports now and return in the correct format.
""")
    ]
    response = llm.invoke(messages)  

    raw_content = response.content

    if "```json" in raw_content:
        json_content = raw_content.split("```json")[1].split("```")[0].strip()
    else:
        json_content = raw_content.strip()

    return json_content

# # Example usage:
# if __name__ == "__main__":
#     r1 = "Patient is a 65-year-old female with hypertension, diabetes. Labs: HbA1c 8.2%, BP 150/95..."
#     r2 = "65F with HTN and DM. Recent labs: A1c 8.2; BP uncontrolled."
#     scores = compare_reports_with_gemini(r1, r2)
#     print(scores)
#     print(type(scores))

#     d = eval(scores)
#     print(d["report1"])

if __name__ == "__main__":
    base_dir = "outputs"
    agentic_dir = os.path.join(base_dir, "agentic")
    nonagentic_dir = os.path.join(base_dir, "non_agentic")
    output_csv = "evaluation_scores.csv"

    agentic_files = [f for f in os.listdir(agentic_dir) if f.startswith("agentic_summary_patient_") and f.endswith(".md")]

    columns = [
        "patient_id",
        "agentic_factual_accuracy", "agentic_clinical_relevance", "agentic_consistency", "agentic_average",
        "nonagentic_factual_accuracy", "nonagentic_clinical_relevance", "nonagentic_consistency", "nonagentic_average"
    ]
    rows = []

    for afile in agentic_files:
        pid = afile.replace("agentic_summary_patient_", "").replace(".md", "")
        nf = f"nonagentic_summary_patient_{pid}.md"
        agentic_path = os.path.join(agentic_dir, afile)
        nonagentic_path = os.path.join(nonagentic_dir, nf)

        print(f"Processing patient {pid}...")

        if not os.path.exists(nonagentic_path):
            print(f"Warning: No matching nonagentic file for patient {pid}")
            continue

        with open(agentic_path, 'r', encoding='utf-8') as f:
            agentic_summary = f.read().strip()
        with open(nonagentic_path, 'r', encoding='utf-8') as f:
            nonagentic_summary = f.read().strip()

        results = compare_reports_with_gemini(agentic_summary, nonagentic_summary)

        scores = eval(results)

        row = {
            "patient_id": pid,
            "agentic_factual_accuracy": scores["report1"]["factual_accuracy"],
            "agentic_clinical_relevance": scores["report1"]["clinical_relevance"],
            "agentic_consistency": scores["report1"]["consistency"],
            "agentic_average": scores["report1"]["average"],
            "nonagentic_factual_accuracy": scores["report2"]["factual_accuracy"],
            "nonagentic_clinical_relevance": scores["report2"]["clinical_relevance"],
            "nonagentic_consistency": scores["report2"]["consistency"],
            "nonagentic_average": scores["report2"]["average"]
        }
        rows.append(row)
        time.sleep(1)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Evaluation completed. Scores saved to {output_csv}")
