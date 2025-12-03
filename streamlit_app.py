import streamlit as st
from main import SemanticSearchEngine
from langchain_core.documents import Document
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env only for local development
load_dotenv()

def get_env(key: str):
    """Read from Streamlit Cloud secrets or local .env automatically."""
    return st.secrets.get(key) or os.getenv(key)

GOOGLE_API_KEY = get_env("GOOGLE_API_KEY")
PINECONE_API_KEY = get_env("PINECONE_API_KEY")

# Page configuration
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
    .stTextArea textarea {
        height: 200px;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    with st.spinner("Initializing search engine..."):
        try:
            st.session_state.engine = SemanticSearchEngine()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.session_state.initialized = False

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# Header
st.title("🔍 Semantic Search Engine")
st.markdown("Powered by Google Gemini Embeddings & Pinecone")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Index info
    if st.session_state.initialized:
        try:
            if st.session_state.engine.vector_store:
                info = st.session_state.engine.get_collection_info()
                st.metric("Total Vectors", info.get('total_vector_count', 0))
            else:
                st.info("No documents indexed yet")
        except Exception as e:
            st.warning("Could not fetch index info")
    
    st.divider()
    
    # Search settings
    st.subheader("Search Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    st.divider()
    
    # Clear index
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.initialized:
            try:
                st.session_state.engine.delete_index()
                st.session_state.engine.vector_store = None
                st.success("Index cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload", "📊 Index Info"])

# TAB 1: SEARCH
with tab1:
    st.header("Search Documents")
    
    if not st.session_state.initialized:
        st.error("Search engine not initialized. Check your API keys.")
    elif not st.session_state.engine.vector_store:
        st.warning("⚠️ No documents indexed yet. Please upload documents first.")
    else:
        # Search form
        with st.form("search_form"):
            query = st.text_input(
                "Enter your search query",
                placeholder="What is deep learning?",
                help="Enter keywords or questions to search"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                search_button = st.form_submit_button("Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_button and query:
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.engine.search_with_scores(query, k=top_k)
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"Search failed: {e}")
        
        # Display results
        if st.session_state.search_results:
            st.subheader(f"📄 Found {len(st.session_state.search_results)} Results")
            
            for idx, (doc, score) in enumerate(st.session_state.search_results, 1):
                with st.container():
                    col1, col2 = st.columns([0.1, 0.9])
                    
                    with col1:
                        st.markdown(f"### {idx}")
                    
                    with col2:
                        # Score badge
                        st.markdown(
                            f'<span class="score-badge">Score: {score:.4f}</span>',
                            unsafe_allow_html=True
                        )
                        
                        # Content
                        st.markdown(f"**Content:**")
                        st.write(doc.page_content)
                        
                        # Metadata
                        if doc.metadata:
                            with st.expander("📋 Metadata"):
                                st.json(doc.metadata)
                    
                    st.divider()

# TAB 2: UPLOAD
with tab2:
    st.header("Upload Documents")
    
    if not st.session_state.initialized:
        st.error("Search engine not initialized. Check your API keys.")
    else:
        upload_type = st.radio(
            "Choose upload method:",
            ["Text Input", "File Upload"],
            horizontal=True
        )
        
        if upload_type == "Text Input":
            st.subheader("📝 Upload Text Content")
            
            with st.form("text_upload_form"):
                title = st.text_input(
                    "Document Title (optional)",
                    placeholder="My Document"
                )
                
                content = st.text_area(
                    "Document Content",
                    placeholder="Paste your text here...",
                    height=300
                )
                
                submit_text = st.form_submit_button("Upload Text", type="primary")
                
                if submit_text and content:
                    with st.spinner("Processing and indexing..."):
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
                            
                            st.success(f"✅ Document uploaded! Created {len(chunks)} chunks.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
        
        else:  # File Upload
            st.subheader("📁 Upload File")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt', 'md'],
                help="Supported formats: PDF, TXT, MD"
            )
            
            if uploaded_file:
                title = st.text_input(
                    "Document Title (optional)",
                    value=uploaded_file.name,
                    key="file_title"
                )
                
                if st.button("Upload File", type="primary"):
                    with st.spinner("Processing and indexing..."):
                        try:
                            # Save file temporarily
                            temp_dir = Path("temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            temp_path = temp_dir / uploaded_file.name
                            
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Load documents
                            docs = st.session_state.engine.load_documents(file_path=str(temp_path))
                            
                            # Update metadata with custom title
                            for doc in docs:
                                doc.metadata['source'] = title
                            
                            # Index
                            chunks = st.session_state.engine.text_splitter.split_documents(docs)
                            
                            if st.session_state.engine.vector_store is None:
                                st.session_state.engine.build_index(docs)
                            else:
                                st.session_state.engine.vector_store.add_documents(chunks)
                            
                            # Cleanup
                            temp_path.unlink()
                            
                            st.success(f"✅ File uploaded! Created {len(chunks)} chunks from {len(docs)} pages.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                            if temp_path.exists():
                                temp_path.unlink()

# TAB 3: INDEX INFO
with tab3:
    st.header("📊 Index Information")
    
    if not st.session_state.initialized:
        st.error("Search engine not initialized.")
    elif not st.session_state.engine.vector_store:
        st.info("No index loaded yet. Upload documents to create an index.")
    else:
        try:
            info = st.session_state.engine.get_collection_info()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Index Name",
                    info.get('index_name', 'N/A')
                )
            
            with col2:
                st.metric(
                    "Total Vectors",
                    info.get('total_vector_count', 0)
                )
            
            with col3:
                st.metric(
                    "Status",
                    "Active" if info.get('total_vector_count', 0) > 0 else "Empty"
                )
            
            st.divider()
            
            st.subheader("Configuration")
            config_data = {
                "Index Name": info.get('index_name', 'N/A'),
                "Vector Count": info.get('total_vector_count', 0),
                "Embedding Model": "gemini-embedding-001",
                "Dimension": "768",
                "Metric": "cosine",
                "Cloud": "AWS",
                "Region": "us-east-1"
            }
            
            for key, value in config_data.items():
                st.text(f"{key}: {value}")
            
        except Exception as e:
            st.error(f"Failed to fetch index info: {e}")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit 🎈 | Powered by Google Gemini & Pinecone</p>
    </div>
""", unsafe_allow_html=True)