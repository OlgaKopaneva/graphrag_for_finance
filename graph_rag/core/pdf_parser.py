import PyPDF2
from typing import List
import os


def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from PDF
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_texts_from_folder(folder_path: str) -> List[str]:
    """
    Parses all PDFs in a folder
    """
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"Parsing {filename}")
            text = extract_text_from_pdf(full_path)
            texts.append(text)
    return texts