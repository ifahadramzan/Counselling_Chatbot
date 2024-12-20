import os
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
PDF_FOLDER = 'pdfs'  # Path to the folder containing all PDFs
PROCESSED_FOLDER = 'ProcessedPDFs'
CHROMA_DB_PATH = 'vectorstore/db_chroma'

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

def read_pdf(file_path):
    """
    Reads and extracts text from a PDF file.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()

def prepare_documents_from_pdf(file_path):
    """
    Converts the extracted PDF text into Document objects.
    """
    text = read_pdf(file_path)
    if not text:
        print(f"No extractable text found in {file_path}. Skipping.")
        return []

    metadata = {"source": file_path}
    document = Document(page_content=text, metadata=metadata)
    return [document]

def vector_embedding(documents):
    """
    Generates embeddings for the given documents and updates the Chroma vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjusted for better granularity
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)

    # Create or load the Chroma vector store
    if os.path.exists(CHROMA_DB_PATH):
        print("Vector store exists, loading and updating...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        vectorstore.add_documents(split_documents)
    else:
        print("Creating a new vector store...")
        vectorstore = Chroma.from_documents(split_documents, embeddings, persist_directory=CHROMA_DB_PATH)

    print("Vector store updated.")

def process_pdfs_in_folder(folder_path):
    """
    Processes all PDFs in the specified folder.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            print(f"Processing: {file_path}")
            try:
                documents = prepare_documents_from_pdf(file_path)
                if documents:
                    vector_embedding(documents)
                    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                    shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file))
                    print(f"Successfully processed and moved: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_pdfs_in_folder(PDF_FOLDER)
    print("All PDFs processed and embeddings updated.")
