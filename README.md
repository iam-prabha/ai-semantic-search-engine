## Semantic Search Engine (Google Gemini + Pinecone)

This project is a minimal semantic search engine built with:

- Google Gemini **embedding-001** model (via `langchain-google-genai`)
- Pinecone as the vector store
- LangChain document loaders and text splitters

### Requirements

- Python 3.13+
- A Google Gemini API key with access to the embeddings API
  - Store it in the environment variable `GOOGLE_API_KEY`
- A Pinecone account with a Vector index configured
  - Store the API key in `PINECONE_API_KEY`

### Setup

1. Install dependencies (using `uv` or `pip`):

```bash
uv sync
# or
pip install -e .
```

2. Set your Google API key:

```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

3. Place your documents (e.g., PDFs) in the `data/` directory.

4. Configure a Pinecone index (example):

   - Index name: `semantic-search`
   - Dimension: `3072` (for `gemini-embedding-001`)
   - Metric: `cosine`

### How embeddings and search work

- `main.py` defines `SemanticSearchEngine`, which:
  - Uses `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`
  - Splits documents into chunks with `RecursiveCharacterTextSplitter`
  - Stores embeddings in a Pinecone index (`semantic-search`)

- `example.py` shows a typical workflow:
  - Loads `data/Fundamentals of Deep Learning.pdf`
  - Builds an index in Pinecone
  - Runs semantic search and prints results

### Google Gemini free-tier limits

Google's free tier for the Gemini API has **strict quotas** for the embeddings endpoint (`embedding-001`), including:

- Requests per minute per project/user
- Requests per day per project/user

If you send too many embedding requests (especially when indexing a large PDF), you can get:

- HTTP `429` errors
- Messages like:
  - `Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests`

### How this project avoids exhausting quota too quickly

1. **Chunking**
   - `RecursiveCharacterTextSplitter` uses a reasonable `chunk_size` to avoid creating too many chunks for each document.

2. **Clear 429 handling**
   - In `main.py`, `build_index` wraps the embedding flow in a `try/except` block catching `GoogleGenerativeAIError`:
   - When a 429 or quota-related error occurs, it prints a clear message explaining:
     - That the Gemini embedding quota has been exceeded
     - Where to check rate limits and usage
     - Suggested mitigations (index fewer chunks, wait for reset, or raise limits)

### If you still see 429 / quota errors

1. **Check your quota and usage**
   - Rate limits and quotas:
     - See the Gemini docs: `https://ai.google.dev/gemini-api/docs/rate-limits`
   - Monitor your usage:
     - `https://ai.dev/usage?tab=rate-limit`

2. **Reduce load further**
   - Use smaller documents or split very large PDFs into multiple files and index them over multiple days.

3. **Wait for daily reset or increase limits**
   - Free-tier daily limits reset once per day.
   - For more intensive usage, consider enabling billing and upgrading your quota.

### Running the example

```bash
uv run streamlit run streamlit_app.py
```

You should see:

- Documents loading and truncation information
- Index-building status
- Collection info (number of embedded documents)
- Semantic search results and similarity scores

If there is a quota problem, you will see a detailed `[Embedding Error]` message with guidance printed from `build_index`.


