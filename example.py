from main import SemanticSearchEngine


def main() -> None:
    # Initialize engine (uses GOOGLE_API_KEY and PINECONE_API_KEY from .env)
    engine = SemanticSearchEngine()

    # Load the PDF document
    print("Loading docs...")
    documents = engine.load_documents(
        file_path="data/Fundamentals of Deep Learning.pdf"
    )

    print("Building index...")
    engine.build_index(documents)

    # Check collection info
    info = engine.get_collection_info()
    print(f"\nCollection info: {info}")

    # Search
    query = "what is deep learning?"
    print(f"\nSearching for: {query}\n")
    results = engine.search(query, k=5)

    # Display documents
    for i, doc in enumerate(results, start=1):
        print(f"{'=' * 80}")
        print(f"Result {i}")
        print(f"{doc.page_content[:500]}...")
        print(f"\nSource: {doc.metadata.get('source', 'unknown')}")
        print(f"{'=' * 80}")

    # Search with scores
    print("\n" + "=" * 80)
    print("SEARCH WITH SCORES")
    print("=" * 80)
    results_with_scores = engine.search_with_scores(query, k=3)
    print(f"\nFound {len(results_with_scores)} results with scores\n")
    
    for i, (doc, score) in enumerate(results_with_scores, start=1):
        print(f"\nResult {i} (Score: {score:.4f})")
        print(f"{doc.page_content[:300]}...")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()