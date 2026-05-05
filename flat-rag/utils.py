import glob
import os
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain as lc_create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def find_files(data_dir: str) -> List[str]:
    """Quét thư mục `data_dir` và trả về list các file .pdf và .txt"""
    patterns = [os.path.join(data_dir, "**", "*.pdf"), os.path.join(data_dir, "**", "*.txt")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(files)


def load_documents(file_paths: List[str]) -> List[Document]:
    """Đọc nội dung từ file .pdf và .txt, trả về list `Document`.

    Mỗi `Document` có `page_content` và `metadata` với khóa `source`.
    """
    docs: List[Document] = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        text = ""
        try:
            if ext == ".pdf":
                reader = PdfReader(path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif ext == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                continue
        except Exception as e:
            # Không dừng cả pipeline nếu một file lỗi
            print(f"Warning: failed to read {path}: {e}")
            continue

        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path}))
    return docs


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Sử dụng RecursiveCharacterTextSplitter để chia văn bản thành các chunk nhỏ.

    Tham số mặc định theo yêu cầu: chunk_size=1000, overlap=200.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_embeddings() -> OpenAIEmbeddings:
    """Tạo embeddings cho Flat RAG.

    Input files -> Text chunks -> Embeddings -> Vector DB.
    """
    return OpenAIEmbeddings()


def create_chroma_from_documents(documents: List[Document], persist_directory: str = ".chromadb") -> Chroma:
    """Tạo embeddings và lưu vào Chroma vector DB local (persist_directory).

    Luồng dữ liệu:
    Input files -> Documents -> Text chunks -> Embeddings -> Chroma (Vector DB)
    """
    embeddings = create_embeddings()

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    try:
        vectordb.persist()
    except Exception:
        # Một vài phiên bản lưu tự động hoặc không expose persist()
        pass
    return vectordb


def create_retrieval_chain(retriever, llm=None):
    """Kết hợp `retriever` và `llm` thành một Retrieval Chain.

    Trả về chain dùng `invoke({"input": question})`.
    """
    if llm is None:
        llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Flat RAG assistant. Answer only from the provided context. "
                "If the context does not contain the answer, say you do not know. "
                "Keep the answer concise and grounded.",
            ),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return lc_create_retrieval_chain(retriever, document_chain)


def build_or_load_vectorstore(data_dir: str = "data", persist_directory: str = ".chromadb", rebuild: bool = False):
    """Toàn bộ flow: quét file -> load -> split -> embeddings -> chroma.

    Nếu `persist_directory` đã tồn tại và chứa DB, bạn có thể load lại (Chroma tự quản lý).
    Trả về `retriever` để dùng cho chain.
    """
    embeddings = create_embeddings()

    if os.path.exists(persist_directory) and not rebuild:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return retriever

    files = find_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}. Place .pdf/.txt files there.")

    raw_docs = load_documents(files)
    print(f"Loaded {len(raw_docs)} source documents")

    chunks = split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    vectordb = create_chroma_from_documents(chunks, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever
