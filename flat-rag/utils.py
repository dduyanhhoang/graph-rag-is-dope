import glob
import os
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain as lc_create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DEFAULT_GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def find_files(data_dir: str) -> List[str]:
    """Quét thư mục `data_dir` và trả về list các file .pdf và .txt."""
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(text.split()),
    )
    return splitter.split_documents(documents)


def get_llm_config() -> dict:
    """Build config for the chat API.

    Chat and embeddings are intentionally configured separately.
    """
    api_key = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("GITHUB_BASE_URL") or DEFAULT_GITHUB_MODELS_BASE_URL
    model = os.getenv("GITHUB_MODEL") or DEFAULT_CHAT_MODEL

    if not api_key:
        raise ValueError(
            "Missing chat API key. Set GITHUB_TOKEN in flat-rag/.env for GitHub Models."
        )

    return {"api_key": api_key, "base_url": base_url, "model": model}


def get_embedding_config() -> dict:
    """Build config for the embedding API.

    This uses a separate API key and endpoint from the chat model.
    """
    api_key = os.getenv("EMBED_API_KEY")
    base_url = os.getenv("EMBED_BASE_URL") or DEFAULT_GITHUB_MODELS_BASE_URL
    model = os.getenv("EMBED_MODEL") or DEFAULT_EMBEDDING_MODEL

    if not api_key:
        raise ValueError(
            "Missing embedding API key. Set EMBED_API_KEY in flat-rag/.env."
        )

    return {"api_key": api_key, "base_url": base_url, "model": model}


def create_embeddings() -> OpenAIEmbeddings:
    """Tạo embeddings cho Flat RAG.

    Input files -> Text chunks -> Embeddings -> Vector DB.
    """
    config = get_embedding_config()
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    return OpenAIEmbeddings(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        chunk_size=batch_size,
    )


def create_chroma_from_documents(documents: List[Document], persist_directory: str = ".chromadb") -> Chroma:
    """Tạo embeddings và lưu vào Chroma vector DB local (persist_directory).

    Luồng dữ liệu:
    Input files -> Documents -> Text chunks -> Embeddings -> Chroma (Vector DB)
    """
    embeddings = create_embeddings()

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    return vectordb


def create_retrieval_chain(retriever, llm=None):
    """Kết hợp `retriever` và `llm` thành một Retrieval Chain.

    Trả về chain dùng `invoke({"input": question})`.
    """
    if llm is None:
        config = get_llm_config()
        llm = ChatOpenAI(
            temperature=0,
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

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


def build_or_load_vectorstore(data_dir: str = "corpus", persist_directory: str = ".chromadb", rebuild: bool = False):
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
