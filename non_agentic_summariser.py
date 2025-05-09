import os
import json
import pandas as pd
import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image, ImageFilter
from llama_cpp import Llama

def extract_from_image(path: str) -> str:
    """OCR-extract text from JPG/PNG/TIFF using Tesseract."""
    img = Image.open(path).convert("L")                                      # grayscale :contentReference[oaicite:0]{index=0}
    img = img.filter(ImageFilter.MedianFilter())                             # denoise :contentReference[oaicite:1]{index=1}
    return pytesseract.image_to_string(img, config="--psm 6")                # OCR config for blocks :contentReference[oaicite:2]{index=2}

def extract_from_pdf(path: str) -> str:
    """Extract raw text from PDF using pdfminer.six."""
    return extract_text(path)                                                # reliable for medical reports :contentReference[oaicite:3]{index=3}

def extract_from_tabular(path: str) -> str:
    """Load CSV/XLSX into DataFrame, then convert to text table."""
    ext = path.lower().split('.')[-1]
    if ext == "csv":
        df = pd.read_csv(path)                                               # CSV reading :contentReference[oaicite:4]{index=4}
    else:
        sheets = pd.read_excel(path, sheet_name=None)                        # Excel reading :contentReference[oaicite:5]{index=5}
        df = pd.concat(sheets.values(), ignore_index=True)
    return df.to_markdown()                                                  # markdown preserves table semantics :contentReference[oaicite:6]{index=6}

def build_context(folder: str) -> str:
    """
    Walk the folder, extract text from each file by extension,
    and concatenate into one context string.
    """
    parts = []
    for root, _, files in os.walk(folder):                                  # recursive folder walk :contentReference[oaicite:7]{index=7}
        for fname in sorted(files):
            path = os.path.join(root, fname)
            ext = fname.lower().split('.')[-1]
            try:
                if ext in ("png","jpg","jpeg","tiff"):
                    text = extract_from_image(path)
                elif ext == "pdf":
                    text = extract_from_pdf(path)
                elif ext in ("csv","xls","xlsx"):
                    text = extract_from_tabular(path)
                else:
                    continue
                parts.append(f"# {fname}\n{text}\n")
            except Exception as e:
                parts.append(f"# {fname}\n[ERROR: {e}]\n")
    return "\n".join(parts)

def summarize_with_openbiollm(context: str, model_path: str, repo_id: str) -> str:
    """
    Send context to OpenBioLLM (Llama.cpp) to generate a structured summary.
    """
    llm = Llama.from_pretrained(                                            # load local GGUF model 
        repo_id=repo_id,
        filename=model_path,
        n_gpu_layers=30,
        n_ctx=8192,
        verbose=False
    )
    prompt = f"""
You are a medical summarization assistant. Generate a concise report with sections:

### Patient Overview
- Demographics
- Chief Complaint

### Clinical Findings
- Labs, Vitals

### Summary & Recommendations

Context:
\"\"\"
{context}
\"\"\"
"""
    out = llm(prompt, max_tokens=1024, stop=[])['choices'][0]['text']        # generate summary :contentReference[oaicite:9]{index=9}
    return out.strip()

def non_agentic_pipeline(input_folder: str, output_folder: str,
                         model_path: str, repo_id: str):
    """End-to-end non-agentic summary: read, concat, summarize, save."""
    ctx = build_context(input_folder)                                       
    summary = summarize_with_openbiollm(ctx, model_path, repo_id)
    os.makedirs(output_folder, exist_ok=True)                                # ensure output dir exists :contentReference[oaicite:10]{index=10}
    out_path = os.path.join(output_folder, "summary.md")
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    # Example usage:
    non_agentic_pipeline(
        input_folder="testcases/patient_1/data",
        output_folder="testcases/patient_1/output",
        model_path="openbiollm-llama3-8b.Q4_K_M.gguf",
        repo_id="aaditya/OpenBioLLM-Llama3-8B-GGUF"
    )

