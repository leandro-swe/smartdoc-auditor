import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import Chunk, RAGEngine

load_dotenv()

st.set_page_config(
    page_title="SmartDoc Auditor",
    page_icon="📄",
    layout="wide",
)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None


def get_server_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def render_sources(sources: list[tuple[Chunk, float]]) -> None:
    if not sources:
        return
    with st.expander("Ver trechos originais recuperados pelo RAG"):
        for rank, (chunk, score) in enumerate(sources, start=1):
            st.markdown(
                f"**#{rank} — {chunk.source} · `[Página {chunk.page}]`** "
                f"· Similaridade de Cosseno: `{score:.4f}`"
            )
            st.caption(chunk.text)
            if rank < len(sources):
                st.divider()


with st.sidebar:
    st.header("Base de Documentos")
    uploaded_files = st.file_uploader(
        "Envie um ou mais arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button(
        "Processar Documentos",
        type="primary",
        use_container_width=True,
    )

    if process_clicked:
        api_key = get_server_api_key()
        if not api_key:
            st.error("Chave GEMINI_API_KEY não configurada no servidor (.env ou Secrets).")
        elif not uploaded_files:
            st.warning("Selecione pelo menos um arquivo PDF.")
        else:
            status_box = st.empty()
            with st.spinner("Processando documentos..."):
                try:
                    engine = st.session_state.rag_engine
                    if engine is None or engine.api_key != api_key:
                        engine = RAGEngine(api_key=api_key)
                    payload = [(f.name, f) for f in uploaded_files]
                    total_chunks = engine.index_pdfs(
                        payload,
                        on_status=status_box.caption,
                    )
                    status_box.empty()
                    st.session_state.rag_engine = engine
                    st.session_state.messages = []
                    st.session_state.index_stats = {
                        "files": len(uploaded_files),
                        "chunks": total_chunks,
                    }
                    st.success(
                        f"{len(uploaded_files)} documento(s) indexado(s) ({total_chunks} trechos)."
                    )
                except Exception as exc:
                    status_box.empty()
                    st.error(f"Falha ao processar documentos: {exc}")

    if st.session_state.index_stats:
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("PDFs", st.session_state.index_stats["files"])
        col2.metric("Chunks", st.session_state.index_stats["chunks"])
        if st.session_state.messages and st.button("Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.title("SmartDoc Auditor")
st.caption(
    "Auditoria e consulta técnica de editais, contratos e regulamentos via RAG com citação exata de páginas."
)

if st.session_state.rag_engine is None:
    st.info(
        "Envie seus documentos PDF na barra lateral e clique em **Processar Documentos** para iniciar a auditoria."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

user_query = st.chat_input(
    "Faça uma pergunta sobre os documentos indexados...",
    disabled=st.session_state.rag_engine is None,
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Auditando trechos relevantes..."):
            try:
                answer_text, retrieved_sources = st.session_state.rag_engine.answer(user_query)
                st.markdown(answer_text)
                render_sources(retrieved_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": retrieved_sources,
                    }
                )
            except Exception as exc:
                st.error(f"Erro ao consultar o modelo: {exc}")

