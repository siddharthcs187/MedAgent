from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import cv2
import pytesseract
from pdfminer.high_level import extract_text
import camelot
import pandas as pd
import numpy as np
import pydicom
import SimpleITK as sitk
from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../.env")

class MedicalCompilerAgent:
    """
    A combined agent for medical data ingestion, summarization via a generalist LLM (e.g., Gemini),
    and specialist medical advice via OpenBioLLM.
    """

    def __init__(
        self,
        gemini_model_name: str = "gemini-1.5-flash",
        # openbio_model_name: str = "aaditya/Llama3-OpenBioLLM-8B",
        device: str = "cpu",
    ):
        # Set up generalist LLM 
        self.gemini_llm = GoogleGenerativeAI(model=gemini_model_name, api_key=os.getenv("GEMINI_API_KEY"))

        # # Set up specialist medical LLM
        # self.tokenizer = AutoTokenizer.from_pretrained(openbio_model_name)
        # self.openbio_model = AutoModelForCausalLM.from_pretrained(
        #     openbio_model_name,
        #     torch_dtype=torch.float16,
        # ).to(device)

        # Prompt templates
        self.summary_template = PromptTemplate(
            input_variables=["context"],
            template=(
                "You are a medical summarization assistant. Generate a concise summary with sections:\n"
                "### Patient Overview\n"
                "- Age, sex, key demographics\n"
                "- Chief complaint & history\n\n"
                "### Clinical Findings\n"
                "- Vital signs & lab trends (include units)\n"
                "- Imaging results\n\n"
                "### Summary & Recommendations\n"
                "- Synthesis\n"
                "- Next steps or treatment considerations\n\n"
                "Context:\n{context}"
            ),
        )
        self.advice_template = PromptTemplate(
            input_variables=["summary"],
            template=(
                "You are a biomedical expert. Given the summary below, provide detailed clinical recommendations and next steps.\n\n"
                "{summary}"
            ),
        )

        # Chains
        # self.summary_chain = LLMChain(
            # llm=self.gemini_llm, prompt=self.summary_template, output_key="summary"
        # )
        self.summary_chain = self.summary_template | self.gemini_llm | StrOutputParser() 
        # self.advice_chain = LLMChain(
        #     llm=OpenAI(model_name=openbio_model_name, temperature=0.2),
        #     prompt=self.advice_template,
        #     output_key="advice",
        # )

        # self.advice_chain = self.advice_template | self.openbio_model | StrOutputParser()
        # self.pipeline = SimpleSequentialChain(
        #     chains=[self.summary_chain, self.advice_chain], verbose=False
        # )

    # --- Ingestion Methods ---
    @staticmethod
    def ocr_image(path: str) -> str:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return pytesseract.image_to_string(thresh, config="--psm 6")

    @staticmethod
    def parse_pdf_text(path: str) -> str:
        return extract_text(path)

    @staticmethod
    def extract_pdf_tables(path: str) -> pd.DataFrame:
        tables = camelot.read_pdf(path, pages="all", flavor="stream")
        # concatenate all tables into one DataFrame (or return list)
        return pd.concat([t.df for t in tables], ignore_index=True)

    @staticmethod
    def load_tabular(path: str) -> pd.DataFrame:
        ext = path.lower().split(".")[-1]
        if ext == "csv":
            return pd.read_csv(path)
        elif ext in ("xls", "xlsx"):
            sheets = pd.read_excel(path, sheet_name=None)
            # concatenate all sheets
            return pd.concat(sheets.values(), ignore_index=True)
        else:
            raise ValueError(f"Unsupported tabular extension: {ext}")

    @staticmethod
    def read_dicom(path: str) -> pd.DataFrame:
        ds = pydicom.dcmread(path)
        if hasattr(ds, "pixel_array"):
            arr = ds.pixel_array
            # flatten pixels into DataFrame for simplicity
            df = pd.DataFrame(arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1))
        else:
            df = pd.DataFrame()
        return df

    @staticmethod
    def preprocess_image(path: str, size=(512, 512)) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        norm = cv2.normalize(
            resized.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX
        )
        return norm

    # --- Pipeline Execution ---
    def run(self, files: list[str]) -> str:
        """
        Ingests a list of file paths, builds context, and returns combined summary + advice.
        """
        context_parts = []
        for f in files:
            ext = f.lower().split(".")[-1]
            if ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
                text = self.ocr_image(f)
                context_parts.append(f"# OCR ({f})\n{text}")
            elif ext == "pdf":
                txt = self.parse_pdf_text(f)
                context_parts.append(f"# PDF Text ({f})\n{txt}")
                tbl = self.extract_pdf_tables(f)
                context_parts.append(f"# PDF Tables ({f})\n{tbl.to_markdown()}")
            elif ext in ("csv", "xls", "xlsx"):
                df = self.load_tabular(f)
                context_parts.append(f"# Tabular ({f})\n{df.to_markdown()}")
            elif ext in ("dcm",):
                df = self.read_dicom(f)
                context_parts.append(f"# DICOM ({f})\n{df.head().to_markdown()}")
            else:
                context_parts.append(f"# Skipped unsupported file type: {f}")

        full_context = "\n\n".join(context_parts)
        # Run the sequential chain: summary then advice
        output = self.pipeline.run({"context": full_context})
        return output

if __name__ == "__main__":
    files = [
    #     "data/note1.png",
    #     "data/report.pdf",
    #     "data/labs.xlsx",
    #     "data/scan.dcm",
    ]
    agent = MedicalCompilerAgent()
    report = agent.run(files)
    print("=== Generated Medical Report ===")
    print(report)
