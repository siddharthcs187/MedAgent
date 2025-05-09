from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from dotenv import load_dotenv
import os

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

# Example usage:
if __name__ == "__main__":
    r1 = "Patient is a 65-year-old female with hypertension, diabetes. Labs: HbA1c 8.2%, BP 150/95..."
    r2 = "65F with HTN and DM. Recent labs: A1c 8.2; BP uncontrolled."
    scores = compare_reports_with_gemini(r1, r2)
    print(scores)
    print(type(scores))

    d = eval(scores)
    print(d["report1"])

