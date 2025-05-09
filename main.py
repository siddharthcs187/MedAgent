import chainlit as cl
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from compiler import MedicalCompilerAgent, MedicalChatAgent

compilerAgent = MedicalCompilerAgent()
chatAgent = MedicalChatAgent()
executor = ThreadPoolExecutor()

@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("uploaded_files", [])
    cl.user_session.set("generated_report", None)

    await cl.Message(content=" Welcome! Please upload all your medical files").send()

    files = await cl.AskFileMessage(
        content="Upload your files (PDF, PNG, CSV). Select multiple or upload one by one!",
        accept=["application/pdf", "image/png", "text/csv"],
        max_files=10,
        max_size_mb=20
    ).send()

    if not files:
        await cl.Message(content="No files uploaded. Please try again.").send()
        return

    uploaded_files = files
    file_list = "\n".join([f"- {file.name}" for file in uploaded_files])
    await cl.Message(content=f"✅ Received {len(uploaded_files)} file(s):\n{file_list}").send()

    action_response = await cl.AskActionMessage(
        content="Click the button to upload",
        actions=[
            cl.Action(name="upload_files", value="upload_files", label="Upload Files", payload={}),
            cl.Action(name="cancel", value="cancel", label="Cancel", payload={}),
        ]
    ).send()

    if action_response.get("value") == "cancel":
        await cl.Message(content="Upload cancelled.").send()
        return

    os.makedirs("uploaded_files", exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_paths.append(file.path)

    await cl.Message(content=f"✅ Saved {len(file_paths)} files. Starting processing...").send()

    generated_report = await asyncio.get_event_loop().run_in_executor(
        executor, compilerAgent.run, file_paths
    )

    report_path = "./summary_report.md"
    cl.user_session.set("generated_report", generated_report)
    cl.user_session.set("report_path", report_path)

    with open(report_path, "w") as f:
        f.write(generated_report)

    await cl.Message(
        content="📄 [Download Medical Report]",
        elements=[
            cl.File(name="Medical_Report.md", path=report_path)
        ]
    ).send()

    await cl.Message(content="Feel free to ask any related questions!").send()

@cl.on_message
async def on_message(msg: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    report = cl.user_session.get("generated_report")
    chat_history.append(msg.content)
    response = chatAgent.run(chat_history, msg.content, report)
    await cl.Message(content=response).send()
    cl.user_session.set("chat_history", chat_history)
