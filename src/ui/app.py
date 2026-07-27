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

    # Sidebar settings
    with st.sidebar:
        st.header("Pipeline Configuration")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=5, value=2)
        
        st.markdown("---")
        st.markdown("**Architecture Overview:**")
        st.markdown("- **Embedding Model:** BAAI/bge-small-en-v1.5")
        st.markdown("- **Vector DB:** Qdrant Local Persistent Store")
        st.markdown("- **LLM Adapter:** Fine-Tuned LoRA (4-bit Compatible)")

        # Sidebar settings
    with st.sidebar:
        st.header("Pipeline Configuration")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=5, value=2)
        
        st.markdown("---")
        if st.button("Reload Pipeline Memory"):
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

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