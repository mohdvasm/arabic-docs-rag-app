from openai import OpenAI 
import base64
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_from_image(file: Path):
    try:
        # Getting the Base64 string
        base64_image = encode_image(file)

        prompt = (
    
        )

        prompt = """ 
        Extract text from the image. Output only the extracted text.
        """

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )

        # print(response.output_text)
        return response.output_text
    except Exception as e:
        print(f"Error while extracting the text from image. {e}")
        return ""
    

def extract_text_from_pdf(file: str):
    try:
        docs = PyMuPDFLoader(file).load()
        return docs
    except Exception as e:
        print(f"Error while loading the text from pdf. {e}")


def create_docs_for_image_data(content: str, filename: str):
    try:
        document = Document(
            page_content=content,
            metadata={
                "source": filename
            }
        )
        return document
    except Exception as e:
        print(f"Error while creating document for image text")