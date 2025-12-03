import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class SemanticSearchEngine:
    def __init__(
        self,
        google_model_name: str = "gemini-embedding-001",
        google_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        pinecone_index_name: str = "semantic-search",
    ):
        """
        Initialize the semantic search engine with Google Gemini embeddings
        and Pinecone as the vector store.

        Args:
            google_model_name: Google Generative AI embedding model name.
            google_api_key: Optional explicit Google API key. If not provided,
                GOOGLE_API_KEY will be read from the environment.
            pinecone_api_key: Optional explicit Pinecone API key. If not
                provided, PINECONE_API_KEY will be read from the environment.
            pinecone_index_name: Name of the Pinecone index to use.
        """
        # Embeddings - Create base embeddings first
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY env var or pass google_api_key.")
        
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model=google_model_name,
            google_api_key=api_key,
            task_type="retrieval_document",
        )
        
        # Test to get actual dimension from base embeddings
        print("Testing embedding dimensions...")
        test_embedding = base_embeddings.embed_query("test")
        actual_dimension = len(test_embedding)
        print(f"✓ Base embeddings produce: {actual_dimension} dimensions")
        
        # Determine expected dimension based on model
        if "gemini-embedding-001" in google_model_name:
            expected_dimension = 3072
        else:
            expected_dimension = actual_dimension
        
        # Check if dimensions match
        if actual_dimension != expected_dimension:
            print(f"⚠ WARNING: Expected {expected_dimension} but got {actual_dimension}")
            print(f"Using actual dimension: {actual_dimension}")
            embedding_dimension = actual_dimension
        else:
            print(f"✓ Dimensions match expected: {expected_dimension}")
            embedding_dimension = expected_dimension
        
        # Use embeddings directly (no wrapper needed if dimensions are correct)
        self.embeddings = base_embeddings

        # Pinecone
        pc_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not pc_key:
            raise ValueError("Set PINECONE_API_KEY env var or pass pinecone_api_key.")

        self.pc = Pinecone(api_key=pc_key)
        self.index_name = pinecone_index_name

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✓ Created Pinecone index: {self.index_name} with dimension {embedding_dimension}")
        else:
            # Verify existing index dimension matches
            print(f"Found existing index '{self.index_name}'")
            index_info = self.pc.describe_index(self.index_name)
            index_dim = index_info.dimension
            
            print(f"Index dimension: {index_dim}")
            print(f"Embedding dimension: {embedding_dimension}")
            
            if index_dim != embedding_dimension:
                raise ValueError(
                    f"\n❌ DIMENSION MISMATCH!\n"
                    f"   Existing index has dimension: {index_dim}\n"
                    f"   But embeddings produce: {embedding_dimension}\n\n"
                    f"Solutions:\n"
                    f"   1. Delete the index at https://app.pinecone.io/\n"
                    f"   2. Or use a different index name\n"
                    f"   3. Or run: engine.delete_index() to delete programmatically"
                )
            print(f"✓ Index dimensions match!")

        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_documents(self, file_path: str = None, directory_path: str = None):
        """
        Load the documents from files or directory.

        Args:
            file_path: path to the single document file (PDF, TXT, etc.)
            directory_path: path to a directory containing documents.
        
        Returns:
            List of document objects.
        """
        documents = []

        if directory_path:
            # load all documents from directory
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
        elif file_path:
            # load single document
            file_extension = Path(file_path).suffix.lower()

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension in (".md", ".txt"):
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
        else:
            raise ValueError("Either file_path or directory_path must be provided")

        return documents
    
    def build_index(self, documents: list[Document]):
        """
        Build vector index from documents using Pinecone.
        """
        try:
            # split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split documents into {len(chunks)} chunks")

            # Build index
            print("Building vector store...")
            self.vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
            )

            print(f"✓ Pinecone index '{self.index_name}' populated with {len(chunks)} vectors.")
            return self.vector_store
            
        except GoogleGenerativeAIError as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                print(
                    "\n❌ [Embedding Error] Google Gemini embedding quota exceeded.\n"
                    "Details from API:\n"
                    f"{msg}\n\n"
                    "How to resolve:\n"
                    "  - Check your Gemini API usage and free-tier quotas:\n"
                    "    https://ai.google.dev/gemini-api/docs/rate-limits\n"
                    "  - Consider indexing fewer pages/chunks per run.\n"
                    "  - Wait for the daily quota reset or enable billing / higher limits.\n"
                )
            else:
                print("\n❌ [Embedding Error] Unexpected Google Generative AI error:\n")
                print(msg)
            raise

    def delete_index(self):
        """
        Delete the current Pinecone index.
        Use this if you need to recreate the index with different dimensions.
        """
        if self.index_name in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.delete_index(self.index_name)
            print(f"✓ Deleted index: {self.index_name}")
        else:
            print(f"ℹ Index {self.index_name} does not exist")

    def load_index(self):
        """
        Connect to an existing Pinecone index.
        """
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
        )
        print(f"✓ Connected to Pinecone index '{self.index_name}'.")
    
    def search(self, query: str, k: int = 5):
        """
        Perform semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Build or load an index first.")
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 5):
        """
        Perform semantic search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (Document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Build or load an index first.")

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_collection_info(self):
        """
        Get information about the current index.

        Returns:
            Dictionary with index information.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")

        # PineconeVectorStore exposes the underlying index via _index
        index = getattr(self.vector_store, "_index", None)
        if index is None:
            return {"index_name": self.index_name, "details": "Index handle not available"}

        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        return {
            "index_name": self.index_name,
            "total_vector_count": total,
        }


def main():
    print("=== Semantic Search Engine with Pinecone ===\n")
    search_engine = SemanticSearchEngine()
    print("\n✓ Engine initialized successfully!")


if __name__ == "__main__":
    main()