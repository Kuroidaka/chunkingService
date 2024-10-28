# import PyPDF2
import pdfplumber


# def extract_text_from_pdf(file):
#     text = ""
#     pdf_reader = PyPDF2.PdfReader(file)
#     for page in pdf_reader.pages:
#         print(page.extract_text)
#         # text += page.extract_text() or ""
#     return text


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text