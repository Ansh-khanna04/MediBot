import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import(
    EURI_API_KEY,
    EURI_BASE_URL, EMBEDDING_MODEL_NAME ,CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_PATH,
)

# Extract text from PDF
def extract_text_from_pdf(pdf_path:str) -> list[dict]:
    """Returns list of {page_num,text} dicts"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # Only include pages with text
            pages.append({"page_num": i, "text": text})
    return pages



def load_all_pdfs_from_dir(dir_path:str) -> list[dict]:
    """Loads every pdf in the folder"""
    all_pages = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(dir_path, filename)
            pages = extract_text_from_pdf(path)
            # tagging all pages with their source file
            for p in pages:
                p["source"] = filename
            all_pages.extend(pages)# it adds all pages into all_pages one by one
            print(f"Loaded {len(pages)} pages from {filename}")
    return all_pages


def split_text_into_chunks(pages:list[dict])-> list[dict]:
    """Splits text into chunks"""
    #Turn raw pages into LangChain Documents with metadata.
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators= ["\n\n", "\n", ".", " ", ""])
    """Try splitting by paragraphs (\n\n)
    If chunk still too big → split by lines (\n)
    Still too big → split by sentences (.)
    Still too big → split by words ( )
    Last resort → split by characters ("")
    """

    all_chunks =[]
    for page in pages:
        docs = splitter.create_documents(
            texts = [page["text"]],
            metadatas = [{"source": page["source"],"page": page["page_num"]}]
        )
        all_chunks.extend(docs)
    print(f"Total chunks created:{len(all_chunks)}")
    return all_chunks
# all chunks is a list containing all chunks inside the Langchain Document class

def build_vector_store(chunks:list[Document]):
    """Builds vector store from chunks and saves it to disk"""
    # EMBEDS + Saves to FAISS
    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    print("Building vector store...")
    vector_store =  FAISS.from_documents(chunks,embeddings)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector Store created and saved to{VECTOR_STORE_PATH}")
    return vector_store

    
def ingest(pdf_dir:str):
    """Main function to ingest PDFs and build vector store"""
    print("Starting ingestion process...")
    print(f"Reading pdfs from {pdf_dir}...")
    pages = load_all_pdfs_from_dir(pdf_dir)
    chunks = split_text_into_chunks(pages)
    build_vector_store(chunks)
    print("Ingestion process completed successfully!")


if __name__ == "__main__":
    ingest("../sample_data")








