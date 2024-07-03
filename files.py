import fitz
import os
from ngram_cosine import preprocess_text

## 
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

##
def load_documents(folder):
    documents = {}
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
            documents[filename] = preprocess_text(file.read())
    return documents