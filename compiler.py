import json
import os
import pandas as pd
import pytesseract
import camelot
import pydicom
from pdfminer.high_level import extract_text
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from llama_cpp import Llama

load_dotenv()


@tool
def ocr_image(path: str) -> str:
    """Extracts text from an image file using OCR."""
    try:
        img = Image.open(path).convert("L")
        img = img.filter(ImageFilter.MedianFilter())
        text = pytesseract.image_to_string(img, config="--psm 6")
        return f"Extracted text from {path}:\n{text}"
    except Exception as e:
        return f"Error extracting text from {path}: {str(e)}"


@tool
def parse_pdf_text(path: str) -> str:
    """Extract raw text from PDF."""
    try:
        text = extract_text(path)
        return f"Extracted text from {path}:\n{text}"
    except Exception as e:
        return f"Error extracting text from {path}: {str(e)}"


@tool
def extract_pdf_tables(path: str) -> str:
    """Extract tables from PDF as markdown."""
    try:
        tables = camelot.read_pdf(path, pages="all", flavor="stream")
        if tables:
            df = pd.concat([t.df for t in tables], ignore_index=True)
            return f"Extracted tables from {path}:\n{df.to_markdown()}"
        else:
            return f"No tables found in {path}."
    except Exception as e:
        return f"Error extracting tables from {path}: {str(e)}"


@tool
def load_tabular(path: str) -> str:
    """Load tabular data (CSV, Excel) as markdown."""
    try:
        ext = path.lower().split(".")[-1]
        if ext == "csv":
            df = pd.read_csv(path)
        else:
            sheets = pd.read_excel(path, sheet_name=None)
            df = pd.concat(sheets.values(), ignore_index=True)
        return f"Loaded tabular data from {path}:\n{df.to_markdown()}"
    except Exception as e:
        return f"Error loading tabular data from {path}: {str(e)}"


@tool
def read_dicom(path: str) -> str:
    """Read and flatten DICOM image data."""
    try:
        ds = pydicom.dcmread(path)
        if hasattr(ds, "pixel_array"):
            arr = ds.pixel_array
            df = pd.DataFrame(
                arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1))
        else:
            df = pd.DataFrame()
        return f"DICOM data preview from {path}:\n{df.head().to_markdown()}"
    except Exception as e:
        return f"Error reading DICOM file {path}: {str(e)}"


def process_tool_call(tool_name: str, tool_args: dict) -> str:
    """Process the tool call based on tool name and arguments."""
    tools = {
        "ocr_image": ocr_image,
        "parse_pdf_text": parse_pdf_text,
        "extract_pdf_tables": extract_pdf_tables,
        "load_tabular": load_tabular,
        "read_dicom": read_dicom
    }

    if tool_name not in tools:
        raise ValueError(f"Unknown tool {tool_name}")

    return tools[tool_name].invoke(tool_args["path"])


class MedicalCompilerAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key)
        self.tools = [ocr_image, parse_pdf_text,
                      extract_pdf_tables, load_tabular, read_dicom]
        self.model_with_tools = self.llm.bind_tools(tools=self.tools)

        self.TOOL_DEFINITIONS = """
        You are a medical file processing assistant. You analyze medical files and extract relevant information.
        
        You have access to the following tools:
        
        1. ocr_image:
           - Extracts text from an image file using OCR.
           - Parameters: path (string)
           - Returns: Extracted text from the image
           
        2. parse_pdf_text:
           - Extract raw text from PDF.
           - Parameters: path (string)
           - Returns: Extracted text from the PDF
           
        3. extract_pdf_tables:
           - Extract tables from PDF as markdown.
           - Parameters: path (string)
           - Returns: Extracted tables from the PDF in markdown format
           
        4. load_tabular:
           - Load tabular data (CSV, Excel) as markdown.
           - Parameters: path (string)
           - Returns: Loaded tabular data in markdown format
           
        5. read_dicom:
           - Read and flatten DICOM image data.
           - Parameters: path (string)
           - Returns: DICOM data preview
        
        Based on the file path, determine the appropriate tool to use and process the file.
        
        Respond in this JSON format:
        ```json
        {
            "thoughts": "Your reasoning for selecting this tool",
            "tool_needed": true/false,
            "tool_name": "name of the tool (if needed)",
            "tool_args": {"path": "file_path"} (if needed),
            "final_answer": "Your analysis of the file" 
        }
        ```
        """

        self.SUMMARY_PROMPT = """
        You are a medical summarization assistant. Generate a concise report with sections:
        
        ### Patient Overview
        - Demographics
        - Complaint
        
        ### Clinical Findings
        - Labs, Vitals
        
        ### Summary & Recommendations
        
        Context:
        {context}
        
        Return a comprehensive medical report based on all information provided.
        """

    def process_file(self, file_path):
        """Process a single file and return the extracted information"""
        try:
            print(f"\n--- Processing file: {file_path} ---")

            # Ask the model to select the appropriate tool
            messages = [
                SystemMessage(content=self.TOOL_DEFINITIONS),
                HumanMessage(content=f"Process this medical file: {file_path}")
            ]

            response = self.model_with_tools.invoke(messages)
            raw_content = response.content
            print("Raw tool selection response:", raw_content)

            # Parse the JSON response
            try:
                # Extract JSON from potentially markdown-wrapped response
                if "```json" in raw_content:
                    json_content = raw_content.split(
                        "```json")[1].split("```")[0].strip()
                else:
                    json_content = raw_content.strip()

                parsed_response = json.loads(json_content)

                if parsed_response.get("tool_needed"):
                    tool_name = parsed_response["tool_name"]
                    tool_args = parsed_response["tool_args"]
                    print(f"Tool to call: {tool_name}\nArguments: {tool_args}")

                    result = process_tool_call(
                        tool_name=tool_name, tool_args=tool_args)

                    # For PDFs, also try to extract tables if we initially used text extraction
                    if tool_name == "parse_pdf_text" and file_path.lower().endswith(".pdf"):
                        try:
                            tables_result = process_tool_call(
                                "extract_pdf_tables", {"path": file_path})
                            if "No tables found" not in tables_result:
                                result += f"\n\n{tables_result}"
                        except Exception as e:
                            print(f"Table extraction failed: {str(e)}")

                    return result
                else:
                    return parsed_response.get("final_answer", f"No suitable tool found for {file_path}")

            except json.JSONDecodeError:
                print("Failed to parse JSON response:", raw_content)
                # Fallback: determine tool based on extension
                ext = file_path.lower().split(".")[-1]
                if ext in ["png", "jpg", "jpeg"]:
                    return process_tool_call("ocr_image", {"path": file_path})
                elif ext == "pdf":
                    return process_tool_call("parse_pdf_text", {"path": file_path})
                elif ext in ["csv", "xlsx", "xls"]:
                    return process_tool_call("load_tabular", {"path": file_path})
                elif ext == "dcm":
                    return process_tool_call("read_dicom", {"path": file_path})
                else:
                    return f"Error: No suitable tool found for {file_path}"

        except Exception as e:
            return f"Error processing {file_path}: {str(e)}"

    def run(self, files):
        """Process multiple files and generate a summary report with quality check"""
        context_parts = []
        for file_path in files:
            result = self.process_file(file_path)
            context_parts.append(f"# {os.path.basename(file_path)}\n{result}")
        full_context = "\n\n".join(context_parts)
        print("\n--- Generating summary report ---")
        
        # Set maximum attempts for report generation
        max_attempts = 3
        quality_threshold = 7.0  # Threshold score out of 10
        
        for attempt in range(1, max_attempts + 1):
            print(f"Report generation attempt {attempt}/{max_attempts}")
            
            # Generate the summary report using the model
            messages = [
                SystemMessage(content=self.SUMMARY_PROMPT.format(context=full_context)),
                HumanMessage(content="Generate a comprehensive medical report based on the provided context.")
            ]
            summary_response = self.llm.invoke(messages).content
            print("Generated report")
            
            # Evaluate report quality using the same LLM
            quality_messages = [
                SystemMessage(content="""You are a medical report quality evaluator. 
                    Rate the following medical report on a scale from 1 to 10 based on:
                    - Comprehensiveness (covers all key medical information)
                    - Clinical relevance (focuses on clinically important findings)
                    - Clarity and organization
                    - Actionability (provides clear next steps)
                    
                    Return ONLY a numeric score between 1 and 10, with no additional text."""),
                HumanMessage(content=f"Please evaluate this medical report and provide a score from 1-10:\n\n{summary_response}")
            ]
            
            quality_response = self.llm.invoke(quality_messages).content
            
            # Extract numeric score from response
            try:
                quality_score = float(quality_response.strip())
                print(f"Report quality score: {quality_score}/10")
            except ValueError:
                # If we can't extract a numeric score, assume it's below threshold to retry
                quality_score = 0
                print("Could not determine quality score, assuming below threshold")
            
            # Check if the quality meets our threshold
            if quality_score >= quality_threshold or attempt == max_attempts:
                break
            
            print(f"Quality score {quality_score} below threshold {quality_threshold}. Regenerating report...")
            
            # For subsequent attempts, add feedback to improve the report
            self.SUMMARY_PROMPT += f"\n\nThe previous attempt scored {quality_score}/10. Please improve the report's comprehensiveness, clinical relevance, clarity, and actionability."
        
        # Generating insights using OpenBioLLM
        llm_insights = Llama.from_pretrained(
            repo_id="aaditya/OpenBioLLM-Llama3-8B-GGUF",
            filename="openbiollm-llama3-8b.Q4_K_M.gguf",
            n_gpu_layers=30,
            n_ctx=8192,
            verbose=False
        )
        
        prompt_insights = f"""<s> You are an experienced clinical decision support assistant developed by Saama AI Labs. Based on the following patient summary, identify the most important actionable clinical insights. These should include recommendations for diagnosis refinement, immediate interventions, further investigations, and potential referrals. Be specific, evidence-based, and avoid vague statements.
        Patient Summary:
        \"\"\"
        {summary_response}
        \"\"\"
        Provide a bullet-point list of actionable clinical insights for this patient.
        """
        
        summary_insights = llm_insights(prompt_insights, max_tokens=512, stop=[])['choices'][0]['text'].strip()
        print("Generated insights")
        
        final_report = f"""
        ==== Patient Summary ====
        {summary_response}
        
        ==== Actionable Clinical Insights ====
        {summary_insights}
        
        ==== Report Quality Information ====
        Report quality score: {quality_score}/10
        Generation attempts: {attempt}/{max_attempts}
        """
        
        return final_report


if __name__ == "__main__":
    files = [
        "/Users/siddharthcs/Desktop/MedAgent/testcases_1/patient_141764/141764_data.csv",
        "/Users/siddharthcs/Desktop/MedAgent/testcases_1/patient_141764/141764_img.png",
        "/Users/siddharthcs/Desktop/MedAgent/testcases_1/patient_141764/141764_report.pdf",
    ]

    agent = MedicalCompilerAgent()
    report = agent.run(files)

    print("\n=== Generated Medical Report ===")
    print(report)

    with open("report.md", "w") as f:
        f.write(report)


class MedicalChatAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key)

    def run(self, chat_history, msg, report):
        chat_history.append(["Human", msg])
        systemPrompt = f"""You are a professional doctor, you have been provided context a summary of the patient's diagnosis. You have also been provided the conversation between you and the patient so far. You are also provided the question the patient has for you. Respond appropriately to the patient's query
        ##HUMAN MESSAGE - {msg}
        ##CONVERSATION HISTORY - {chat_history}
        ##PATIENT DIAGNOSIS - {report}
        """
        messages = [
            SystemMessage(content=systemPrompt),
            HumanMessage(content=msg)
        ]
        response = self.llm.invoke(messages).content
        chat_history.append(["System", response])
        return response
