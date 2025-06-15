import gradio as gr
import random
from arabic_rag import (
    vector_store,
    update_vectorstore,
    graph, 
)

# Upload files and add into knowledge base
def upload_file(files):
    file_paths = [file.name for file in files]
    update_vectorstore(file_paths)
    gr.Info('Knowledge base is updated')
    return file_paths


def generate_response(message, history):

    # If files added into chat to upload into knowledge base or ask current docuement content
    if message['files']:
        update_vectorstore(message['files'])

    response = graph.invoke({"question": message['text']})
    return response['answer']

with gr.Blocks() as demo:

    # Tab for Chatting
    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=generate_response, 
            type="messages",
            multimodal=True
        )

    # Tab of uploading files for knowledge base
    with gr.Tab("Build knowledge base"):
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "pdf"], file_count="multiple")
        upload_button.upload(upload_file, upload_button, file_output)


if __name__ == "__main__":
    demo.launch()