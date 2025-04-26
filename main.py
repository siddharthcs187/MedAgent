import chainlit as cl
import os
from compiler import MedicalCompilerAgent

agent = MedicalCompilerAgent()

uploaded_files = []

@cl.on_chat_start
async def start():
    await cl.Message(content=" Welcome! Please upload all your medical files").send()

    files = await cl.AskFileMessage(
        content="Upload your files (PDF, PNG, CSV). Select multiple or upload one by one!",
        accept=["application/pdf", "image/png", "text/csv"],
        max_files=10,
        max_size_mb=20,
        timeout=300
    ).send()

    if not files:
        await cl.Message(content="No files uploaded. Please try again.").send()
        return

    uploaded_files.extend(files)  # ✅ It's a list of AskFileResponse
    file_list = "\n".join([f"- {file.name}" for file in uploaded_files])
    await cl.Message(content=f"✅ Received {len(uploaded_files)} file(s):\n{file_list}").send()

    action_response = await cl.AskActionMessage(
        content="Click the button to upload",
        actions=[
            cl.Action(name="upload_files", value="upload_files", label="Upload Files", payload={}),
            cl.Action(name="cancel", value="cancel", label="Cancel", payload={}),
        ],
        timeout=300,
    ).send()

    if action_response.get("value") == "cancel":
        await cl.Message(content="Upload cancelled.").send()
        return

    os.makedirs("uploaded_files", exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_paths.append(file.path)

    await cl.Message(content=f"✅ Saved {len(file_paths)} files. Starting processing...").send()

    report = agent.run(file_paths)

    report_path = "./summary_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    await cl.Message(content="✅ Summary complete! Here's a preview:").send()
    await cl.Message(content=f"```markdown\n{report[:4000]}\n```").send()
    await cl.File(name="Medical_Report.md", path=report_path).send(for_id=cl.context.current_step.id)
    action = await cl.AskActionMessage(
        content="📄 Click the button below to download your medical report!",
        actions=[
            cl.Action(name="download_report", value="download_report", label="Download Report", payload={}),
        ],
        timeout=300,
    ).send()

    if action.get("value") == "download_report":
        await cl.File(name="Medical_Report.md", path=report_path).send(for_id=cl.context.current_step.id)
