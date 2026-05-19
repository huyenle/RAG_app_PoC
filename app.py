import streamlit as st

from rag import BasicRAGAdapter, LightRAGAdapter

RAG_OPTIONS = {
    "Basic RAG (FAISS)": BasicRAGAdapter,
    "Graph RAG (LightRAG)": LightRAGAdapter,
}

RAG_ICONS = {
    "BasicRAGAdapter": "🔍",
    "LightRAGAdapter": "🕸️",
}

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")

# Store per-file adapters and chat history in session state
if "adapters" not in st.session_state:
    st.session_state["adapters"] = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Sidebar: Upload PDFs ---
with st.sidebar:
    st.header("Settings")
    rag_choice = st.selectbox("RAG method", list(RAG_OPTIONS.keys()))

    st.divider()
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process PDFs"):
        new_count = 0
        adapter_class = RAG_OPTIONS[rag_choice]
        for uploaded in uploaded_files:
            if uploaded.name in st.session_state["adapters"]:
                continue  # skip already processed
            path = f"/tmp/{uploaded.name}"
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner(f"Processing {uploaded.name}..."):
                new_adapter = adapter_class()
                new_adapter.index_document(path)
                st.session_state["adapters"][uploaded.name] = new_adapter
            new_count += 1

        if new_count:
            st.success(f"Processed {new_count} new PDF(s).")
        else:
            st.info("All files already processed.")

    # --- Select which document to query ---
    st.divider()
    doc_names = list(st.session_state["adapters"].keys())
    if doc_names:
        selected_doc = st.selectbox("Select document to query", doc_names)
        st.session_state["selected_doc"] = selected_doc
        adapter = st.session_state["adapters"][selected_doc]
        st.caption(f"Using: {type(adapter).__name__}")
        if st.button("Reprocess selected document"):
            with st.spinner(f"Reprocessing {selected_doc}..."):
                path = f"/tmp/{selected_doc}"
                adapter = RAG_OPTIONS[rag_choice]()
                adapter.index_document(path)
                st.session_state["adapters"][selected_doc] = adapter
            st.success(f"Reprocessed {selected_doc}.")
            st.rerun()
    else:
        st.info("Upload and process a PDF to get started.")

# --- Chat ---
for msg in st.session_state["messages"]:
    icon = msg.get("icon")
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "selected_doc" not in st.session_state or not st.session_state["adapters"]:
        response = "Please upload and process a PDF first."
        elapsed = 0
    else:
        import time
        doc_name = st.session_state["selected_doc"]
        rag_adapter = st.session_state["adapters"][doc_name]
        start = time.time()
        response = rag_adapter.query(prompt)
        elapsed = time.time() - start

    adapter_name = type(rag_adapter).__name__ if "rag_adapter" in dir() else ""
    icon = RAG_ICONS.get(adapter_name, "🤖")
    st.session_state["messages"].append({"role": "assistant", "content": response, "icon": icon})
    with st.chat_message("assistant", avatar=icon):
        st.markdown(response)
        if elapsed:
            st.caption(f"Response time: {elapsed:.1f}s")
