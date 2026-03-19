from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

dir = Path(r'C:\Users\Felipe\Documents\Python\Maat\base')

# Carrega o PDF
loader = PyPDFLoader(dir / "REGULAMENTO_GERAL_APJ_GOB.pdf")
pages = loader.load()

#print(f"Total de páginas: {len(pages)}")
#print(f"\nTrecho da primeira página:\n{pages[0].page_content[:300]}")

# Dividir arquivo em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 150
)

chunks = splitter.split_documents(pages)

#print(f"\nTotal de chunks: {len(chunks)}")
#print(f"\nExemplo de chunk: {chunks[0].page_content}")

# Criação de modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name ="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Transformar os chunks em vetores e salvar no FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print("Índice vetorial salvo com sucesso!")