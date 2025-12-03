import streamlit as st
from main import SemanticSearchEngine
from langchain_core.documents import Document
import pandas as pd
import plotly.express as px
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    with st.spinner("🚀 Initializing search engine..."):
        try:
            st.session_state.engine = SemanticSearchEngine()
            st.session_state.initialized = True
            st.session_state.upload_history = []
            st.session_state.search_history = []
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.session_state.initialized = False

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔍 Semantic Search Engine")
    st.markdown("*AI-powered document search with Google Gemini*")

with col2:
    if st.session_state.initialized:
        st.success("✅ Ready")
    else:
        st.error("❌ Not Ready")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
    st.header("Control Panel")
    
    # Index stats
    if st.session_state.initialized and st.session_state.engine.vector_store:
        try:
            info = st.session_state.engine.get_collection_info()
            vector_count = info.get('total_vector_count', 0)
            
            st.metric("📊 Total Vectors", vector_count)
            st.metric("📁 Documents Uploaded", len(st.session_state.upload_history))
            st.metric("🔍 Searches Made", len(st.session_state.search_history))
        except:
            st.warning("⚠️ Could not fetch stats")
    else:
        st.info("💡 Upload documents to get started")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Search Settings")
    top_k = st.slider("Results to show", 1, 20, 5)
    show_metadata = st.checkbox("Show metadata", value=True)
    show_scores = st.checkbox("Show similarity scores", value=True)
    
    st.divider()
    
    # Actions
    st.subheader("🛠️ Actions")
    if st.button("🗑️ Clear Index", type="secondary", use_container_width=True):
        if st.session_state.initialized:
            try:
                st.session_state.engine.delete_index()
                st.session_state.engine.vector_store = None
                st.session_state.upload_history = []
                st.success("✅ Index cleared!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {e}")
    
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📤 Upload", "📊 Analytics", "ℹ️ About"])

# TAB 1: SEARCH
with tab1:
    st.header("Search Documents")
    
    if not st.session_state.initialized:
        st.error("🚫 Search engine not initialized")
    elif not st.session_state.engine.vector_store:
        st.warning("⚠️ No documents indexed. Upload documents first!")
    else:
        # Search interface
        query = st.text_input(
            "🔎 What are you looking for?",
            placeholder="Enter your search query...",
            key="search_input"
        )
        
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🧹 Clear Results", use_container_width=True)
        
        if clear_btn:
            st.session_state.search_results = []
            st.rerun()
        
        # Perform search
        if search_btn and query:
            with st.spinner("🔎 Searching..."):
                try:
                    start_time = time.time()
                    results = st.session_state.engine.search_with_scores(query, k=top_k)
                    search_time = time.time() - start_time
                    
                    st.session_state.search_results = results
                    st.session_state.search_history.append({
                        'query': query,
                        'results': len(results),
                        'time': search_time
                    })
                    
                    st.success(f"✅ Found {len(results)} results in {search_time:.2f}s")
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")
        
        # Display results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.divider()
            st.subheader(f"📄 Search Results ({len(st.session_state.search_results)})")
            
            for idx, (doc, score) in enumerate(st.session_state.search_results, 1):
                with st.container():
                    col1, col2 = st.columns([0.05, 0.95])
                    
                    with col1:
                        st.markdown(f"### {idx}")
                    
                    with col2:
                        # Score
                        if show_scores:
                            score_color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
                            st.markdown(
                                f'<span style="background: {score_color}; color: white; padding: 0.2rem 0.8rem; '
                                f'border-radius: 15px; font-size: 0.85rem; font-weight: 600;">'
                                f'Score: {score:.4f}</span>',
                                unsafe_allow_html=True
                            )
                        
                        # Content
                        st.markdown("**Content:**")
                        st.write(doc.page_content)
                        
                        # Metadata
                        if show_metadata and doc.metadata:
                            with st.expander("📋 View Metadata"):
                                st.json(doc.metadata)
                        
                        # Copy button
                        if st.button(f"📋 Copy", key=f"copy_{idx}"):
                            st.code(doc.page_content, language=None)
                    
                    st.divider()

# TAB 2: UPLOAD
with tab2:
    st.header("Upload Documents")
    
    if not st.session_state.initialized:
        st.error("🚫 Search engine not initialized")
    else:
        upload_method = st.radio(
            "Choose upload method:",
            ["📝 Text Input", "📁 File Upload", "📂 Bulk Upload"],
            horizontal=True
        )
        
        # Text Input
        if upload_method == "📝 Text Input":
            with st.form("text_form"):
                title = st.text_input("📌 Title", placeholder="My Document")
                content = st.text_area("📄 Content", height=300, placeholder="Paste your text here...")
                
                submitted = st.form_submit_button("📤 Upload", type="primary", use_container_width=True)
                
                if submitted and content:
                    with st.spinner("⚙️ Processing..."):
                        try:
                            doc = Document(
                                page_content=content,
                                metadata={"source": title or "Text Input", "type": "text"}
                            )
                            
                            chunks = st.session_state.engine.text_splitter.split_documents([doc])
                            
                            if st.session_state.engine.vector_store is None:
                                st.session_state.engine.build_index([doc])
                            else:
                                st.session_state.engine.vector_store.add_documents(chunks)
                            
                            st.session_state.upload_history.append({
                                'title': title or "Text Input",
                                'chunks': len(chunks),
                                'type': 'text'
                            })
                            
                            st.success(f"✅ Uploaded! Created {len(chunks)} chunks")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")
        
        # File Upload
        elif upload_method == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt', 'md'],
                help="Supported: PDF, TXT, MD"
            )
            
            if uploaded_file:
                title = st.text_input("📌 Title (optional)", value=uploaded_file.name)
                
                if st.button("📤 Upload File", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Processing file..."):
                        try:
                            temp_dir = Path("temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            temp_path = temp_dir / uploaded_file.name
                            
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            docs = st.session_state.engine.load_documents(file_path=str(temp_path))
                            
                            for doc in docs:
                                doc.metadata['source'] = title
                            
                            chunks = st.session_state.engine.text_splitter.split_documents(docs)
                            
                            if st.session_state.engine.vector_store is None:
                                st.session_state.engine.build_index(docs)
                            else:
                                st.session_state.engine.vector_store.add_documents(chunks)
                            
                            temp_path.unlink()
                            
                            st.session_state.upload_history.append({
                                'title': title,
                                'chunks': len(chunks),
                                'type': 'file'
                            })
                            
                            st.success(f"✅ Uploaded! {len(chunks)} chunks from {len(docs)} pages")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")
        
        # Bulk Upload
        else:
            st.info("📂 Upload multiple files at once")
            files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            if files and st.button("📤 Upload All", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(files):
                    status_text.text(f"Processing {file.name}...")
                    
                    try:
                        temp_dir = Path("temp_uploads")
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / file.name
                        
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        docs = st.session_state.engine.load_documents(file_path=str(temp_path))
                        chunks = st.session_state.engine.text_splitter.split_documents(docs)
                        
                        if st.session_state.engine.vector_store is None:
                            st.session_state.engine.build_index(docs)
                        else:
                            st.session_state.engine.vector_store.add_documents(chunks)
                        
                        temp_path.unlink()
                        
                        st.session_state.upload_history.append({
                            'title': file.name,
                            'chunks': len(chunks),
                            'type': 'file'
                        })
                        
                    except Exception as e:
                        st.error(f"Failed to process {file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(files))
                
                status_text.text("✅ All files processed!")
                time.sleep(1)
                st.rerun()

# TAB 3: ANALYTICS
with tab3:
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.upload_history:
        st.info("📭 No data yet. Upload documents to see analytics.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Total Uploads", len(st.session_state.upload_history))
        
        with col2:
            total_chunks = sum(item['chunks'] for item in st.session_state.upload_history)
            st.metric("🧩 Total Chunks", total_chunks)
        
        with col3:
            st.metric("🔍 Total Searches", len(st.session_state.search_history))
        
        st.divider()
        
        # Upload history
        if st.session_state.upload_history:
            st.subheader("📚 Upload History")
            df = pd.DataFrame(st.session_state.upload_history)
            st.dataframe(df, use_container_width=True)
            
            # Chart
            fig = px.bar(df, x='title', y='chunks', title='Chunks per Document')
            st.plotly_chart(fig, use_container_width=True)
        
        # Search history
        if st.session_state.search_history:
            st.divider()
            st.subheader("🔍 Recent Searches")
            df_search = pd.DataFrame(st.session_state.search_history[-10:])
            st.dataframe(df_search, use_container_width=True)

# TAB 4: ABOUT
with tab4:
    st.header("ℹ️ About")
    
    st.markdown("""
    ### 🔍 Semantic Search Engine
    
    This application uses advanced AI technology to perform semantic search on your documents.
    
    **Features:**
    - 🤖 Google Gemini embeddings (768 dimensions)
    - 📊 Pinecone vector database
    - 🔍 Similarity-based search
    - 📄 PDF, TXT, MD support
    - ⚡ Fast and accurate
    
    **Technology Stack:**
    - **Frontend:** Streamlit
    - **Embeddings:** Google Gemini AI
    - **Vector DB:** Pinecone
    - **Framework:** LangChain
    
    **How it works:**
    1. Upload your documents
    2. Documents are split into chunks
    3. Each chunk is converted to a 768-dim vector
    4. Vectors are stored in Pinecone
    5. Search queries are also vectorized
    6. Most similar chunks are returned
    
    ---
    Made with ❤️ using Streamlit
    """)