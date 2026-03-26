# ⚖️ Maat — Assistente Inteligente de Regulamentos

Maat é uma aplicação em Python com interface em Streamlit que utiliza **RAG (Retrieval-Augmented Generation)** para responder perguntas com base no Regulamento Geral da APJ/GOB.

A solução combina:
- Busca vetorial (FAISS);
- Busca semântica + lexical (BM25);
- Expansão de query com LLM;
- Interface conversacional.

## 🚀 Demonstração

Interface estilo chat com:
- Histórico de conversa;
- Respostas baseadas **exclusivamente no documento**;
- Tratamento de rate limit;
- Background personalizado.

## 🧠 Como funciona

O fluxo da aplicação segue:

1. Usuário faz uma pergunta  
2. A pergunta é expandida via LLM (sinônimos e variações)  
3. A busca é feita em dois métodos:  
   - FAISS (semântica)  
   - BM25 (palavra-chave)  
4. Os resultados são combinados e deduplicados  
5. O contexto é enviado ao modelo  
6. O modelo responde **somente com base no conteúdo encontrado**

## 🏗️ Estrutura do Projeto

```
MAAT/
│
├── base/                  # Documento fonte (PDF)
├── faiss_index/           # Índice vetorial salvo
│   ├── index.faiss
│   └── index.pkl
│
├── imagens/
│   └── maat.png           # Background da aplicação
│
├── .streamlit/            # Configurações do Streamlit
├── .vscode/               # Configurações do VSCode
│
├── Maat.py                # Aplicação principal (Streamlit + RAG)
├── rag.py                 # Script para gerar o índice vetorial
├── requirements.txt       # Dependências
└── .gitignore
```

## 🚀 Como rodar localmente

**1. Clone o repositório**
```bash
git clone https://github.com/felipe-roch/Maat.git
cd Maat
```

**2. Instale as dependências**
```bash
pip install -r requirements.txt
```

**3. Configure sua chave da Groq**

Crie o arquivo `.streamlit/secrets.toml` na raiz do projeto:
```toml
GROQ_API_KEY = "sua_chave_aqui"
```

**4. Rode a aplicação**
```bash
streamlit run Maat.py
```

## 🔑 Deploy no Streamlit Cloud

Na tela de configuração do app, acesse **Advanced settings → Secrets** e adicione:
```toml
GROQ_API_KEY = "sua_chave_aqui"
```

## Autor

Felipe da Rocha
[Linkedin](https://www.linkedin.com/in/felipedarochaferreira/) · [GitHub](http://github.com/felipe-roch)