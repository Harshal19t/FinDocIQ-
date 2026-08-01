import sys
import os
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.rag_chain import FinancialRAGPipeline


@st.cache_resource
def load_rag_pipeline():
    """Cache the RAG pipeline in memory to prevent reloading on user interactions."""
    return FinancialRAGPipeline()


def main():
    st.set_page_config(
        page_title="Financial AI Assistant",
        layout="wide"
    )

    st.title("Financial Statement RAG Assistant")
    st.caption("Powered by Qwen2.5-1.5B (LoRA Fine-Tuned) & Qdrant Vector Store")

    # --- SIDEBAR LAYOUT ---
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.caption("Financial RAG Assistant Configuration")
        st.divider()

        # Pipeline Settings
        st.subheader("Pipeline Configuration")
        top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of document context chunks retrieved per query.",
            key="sidebar_top_k_slider"
        )

        st.divider()

        # Architecture Overview
        st.subheader("Architecture Overview")
        st.markdown(
            """
            - **Embedding Model:** `BAAI/bge-small-en-v1.5`
            - **Vector DB:** Qdrant Local Store
            - **LLM Adapter:** Fine-Tuned Qwen2.5 (LoRA)
            """
        )

        st.divider()

        # Actions & Controls
        st.subheader("System Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Reload Pipeline", key="btn_reload_pipeline", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()

        with col2:
            if st.button("Clear Chat", key="btn_clear_chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # --- MAIN CHAT UI ---
    # Initialize chat session history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load RAG pipeline with spinner
    with st.spinner("Initializing LLM and Vector Store..."):
        try:
            rag_chain = load_rag_pipeline()
        except Exception as e:
            st.error(f"Failed to load RAG pipeline: {e}")
            return

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Cited Sources"):
                    for src in message["sources"]:
                        st.write(f"- **Page {src['page']}** (ID: `{src['chunk_id']}`) | Similarity Score: `{src['score']}`")

    # Handle user query input
    if prompt := st.chat_input("Ask a question about financial statements..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base & generating answer..."):
                result = rag_chain.query(prompt, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("View Cited Sources"):
                        for src in sources:
                            st.write(f"- **Page {src['page']}** (ID: `{src['chunk_id']}`) | Similarity Score: `{src['score']}`")

        # Save assistant message to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()