import re
import os
import base64
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Carrega o FAISS uma única vez (não recarrega a cada interação)
@st.cache_resource
def carregar_rag():
    BASE_DIR = Path(__file__).parent
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vectorstore = FAISS.load_local(
        BASE_DIR / "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    cliente = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    return vectorstore, cliente

vectorstore, cliente = carregar_rag()

def expandir_query(pergunta, cliente):
    resposta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Você é um assistente especialista em regulamentos da APJ/GOB.
Dada uma pergunta, reescreva ela expandindo com sinônimos, termos relacionados 
e variações que possam aparecer num regulamento oficial.
Retorne APENAS a pergunta expandida, sem explicações."""
            },
            {
                "role": "user",
                "content": f"Pergunta original: {pergunta}"
            }
        ]
    )
    return resposta.choices[0].message.content

def carregar_imagem_base64(caminho):
    with open(caminho, "rb") as f:
        dados = f.read()
    extensao = caminho.split(".")[-1].lower()
    tipo = "jpeg" if extensao == "jpg" else extensao
    return f"data:image/{tipo};base64,{base64.b64encode(dados).decode()}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_maat = carregar_imagem_base64(os.path.join(BASE_DIR, "imagens", "maat.png"))

st.set_page_config(
    page_title="Deusa Maat",
    page_icon="⚖️",
    layout="centered"
)

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{img_maat}");
        background-size: contain;
        background-position: left center;
        background-repeat: no-repeat;
        background-color: #fefefe;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    .stChatFloatingInputContainer {{
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    h1 {{
        color: #3b1f0a !important;
        font-family: Georgia, serif !important;
        text-align: center;
        margin-top: -4rem !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }}
    .st-emotion-cache-128upt6 {{
        background-color: transparent !important;
        background: transparent !important;
    }}
    [data-testid="stCaptionContainer"] {{
        text-align: center !important;
    }}
    .stChatMessage p {{
        color: #2b1200 !important;
        font-family: Georgia, serif !important;
        font-size: 1rem;
        line-height: 1.6;
    }}
    [data-testid="stSidebar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Maat, a Deusa da Justiça")
st.caption("Assistente de consulta ao Regulamento Geral da APJ/GOB")

# Verifica bloqueio por rate limit
if "bloqueado_ate" in st.session_state:
    import time
    restante = int(st.session_state["bloqueado_ate"] - time.time())
    if restante > 0:
        st.error(f"⏳ Maat está em repouso. Volte em {restante // 60}min {restante % 60}s.")
        if st.button("🔄 Verificar se já posso perguntar"):
            st.rerun()
        st.stop()
    else:
        del st.session_state["bloqueado_ate"]

if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []

# Labels nas mensagens do histórico
for msg in st.session_state["mensagens"]:
    with st.chat_message(msg["role"]):
        label = "Maat" if msg["role"] == "assistant" else "Você"
        st.write(f"**{label}:** {msg['content']}")

pergunta = st.chat_input("Digite sua pergunta sobre o regulamento...")

if pergunta:
    with st.chat_message("user"):
        st.write(f"**Você:** {pergunta}")       # ← ADICIONADO: label
    st.session_state["mensagens"].append({"role": "user", "content": pergunta})

    query_expandida = expandir_query(pergunta, cliente)

    all_docs = list(vectorstore.docstore._dict.values())
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    faiss_docs = faiss_retriever.invoke(query_expandida)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5
    bm25_docs = bm25_retriever.invoke(query_expandida)

    vistos = set()
    resultados = []
    for doc in faiss_docs + bm25_docs:
        if doc.page_content not in vistos:
            vistos.add(doc.page_content)
            resultados.append(doc)

    contexto = "\n\n".join([doc.page_content for doc in resultados])

    # try/except com tratamento de rate limit
    try:
        with st.spinner("Consultando o regulamento..."):
            resposta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um assistente especialista no regulamento da APJ/GOB.
Responda a pergunta do usuário baseando-se APENAS no contexto fornecido.
Se a resposta não estiver no contexto, diga que não encontrou a informação no documento.
Quando citar artigos ou parágrafos, use APENAS a numeração que aparecer explicitamente 
no trecho — nunca infira ou invente numeração."""
                    },
                    {
                        "role": "user",
                        "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
                    }
                ]
            )
            texto_resposta = resposta.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(f"**Maat:** {texto_resposta}")    # ← ADICIONADO: label
        st.session_state["mensagens"].append({"role": "assistant", "content": texto_resposta})

    except Exception as e:
        import time
        mensagem_erro = str(e)
        segundos = 300
        match = re.search(r'try again in (\d+)m(\d+)', mensagem_erro)
        if match:
            segundos = int(match.group(1)) * 60 + int(match.group(2))
        st.session_state["bloqueado_ate"] = time.time() + segundos
        st.warning(f"Maat precisa descansar. Tente novamente em {segundos // 60}min {segundos % 60}s.")
        st.rerun()