import os
from typing import List
from compiler import MedicalCompilerAgent
from non_agentic_summariser import non_agentic_pipeline

def generate_reports(
    testcases_root: str,
    n: int,
    agentic_output_dir: str,
    non_agentic_output_dir: str,
    model_path: str,
    repo_id: str
):
    """
    Walk through `testcases_root`, pick first n patient folders,
    run both agentic and non-agentic pipelines, and save outputs.
    """
    all_entries = os.listdir(testcases_root)                                              
    patient_folders = [
        name for name in sorted(all_entries)
        if os.path.isdir(os.path.join(testcases_root, name))
    ]                                                                                     
    selected = patient_folders[:n]                                                        

    os.makedirs(agentic_output_dir, exist_ok=True)                                        
    os.makedirs(non_agentic_output_dir, exist_ok=True)

    agent = MedicalCompilerAgent()  

    for patient in selected:
        patient_data_folder = os.path.join(testcases_root, patient)
        files: List[str] = []

        for fname in sorted(os.listdir(patient_data_folder)):                             
            ext = fname.lower().split('.')[-1]
            if ext in ("pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "dcm"):
                files.append(os.path.join(patient_data_folder, fname))

        agentic_report = agent.run(files)
        agentic_path = os.path.join(
            agentic_output_dir, f"agentic_summary_{patient}.md"
        )

        with open(agentic_path, "w") as f:
            f.write(agentic_report)
        print(f"Wrote agentic report: {agentic_path}")

        non_agentic_summary = non_agentic_pipeline(
            input_folder=patient_data_folder,
            output_folder=non_agentic_output_dir,
            model_path=model_path,
            repo_id=repo_id
        )

if __name__ == "__main__":
    generate_reports(
        testcases_root="testcases_2",
        n=5,
        agentic_output_dir="outputs/agentic",
        non_agentic_output_dir="outputs/non_agentic",
        model_path="openbiollm-llama3-8b.Q4_K_M.gguf",
        repo_id="aaditya/OpenBioLLM-Llama3-8B-GGUF"
    )
