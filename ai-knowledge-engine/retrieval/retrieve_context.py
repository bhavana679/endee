"""
Retrieval Module
Performs semantic similarity search against the Endee Vector database to return relevant context.
"""
import os
import sys
import json
import argparse

# Add the project root to the system path to allow importing internal modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import google.generativeai as genai
from retrieval.endee_client import EndeeClient
from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, ENDEE_HOST, ENDEE_PORT

class ContextRetriever:
    """
    Handles dense vector generation for incoming queries and returns 
    top k matching document context strings.
    """
    def __init__(self, index_name: str = "ai_knowledge"):
        self.endee_client = EndeeClient(host=ENDEE_HOST, port=int(ENDEE_PORT))
        self.index_name = index_name
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model_name = EMBEDDING_MODEL_NAME
        self.vector_dim = EMBEDDING_DIMENSION

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Takes a raw user query string, converts to an embedding,
        and retrieves top matching chunks from Endee.
        """
        # Step 1: Embed query string via Gemini
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']

        # Step 2: Query the DB
        raw_results = self.endee_client.search(
            index_name=self.index_name, 
            query_vector=query_embedding,
            k=top_k
        )
        
        # Step 3: Parse Endee MetaData correctly
        structured_results = []
        for idx, res in enumerate(raw_results, 1):
            if not isinstance(res, list) or len(res) < 3:
                continue
                
            distance = res[0]
            vector_id = res[1]
            meta_raw = res[2]
            
            try:
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode('utf-8')
                meta_json = json.loads(meta_raw)
                
                structured_results.append({
                    "rank": idx,
                    "id": vector_id,
                    "distance": round(distance, 4),
                    "text": meta_json.get("text", "No text provided"),
                    "source": meta_json.get("source", "Unknown")
                })
            except Exception as e:
                print(f"Error parsing metadata for result '{vector_id}': {str(e)}")
                
        return structured_results


if __name__ == "__main__":
    # Provides testing purely through the Command Line
    parser = argparse.ArgumentParser(description="Endee Vector Db Context Retrieval Engine")
    parser.add_argument("query", type=str, help="The query text you wish to search for semantically.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return.")
    
    args = parser.parse_args()
    
    retriever = ContextRetriever()
    
    print(f"Semantic searching for: '{args.query}'...\n")
    results = retriever.search(args.query, top_k=args.k)
    
    if not results:
        print("No matching results found or index is missing.")
    else:
        print("## Retrieval Diagnostics\n")
        for item in results:
            print(f"Rank {item['rank']} | {item['source']} | distance: {item['distance']}")
        
        print("\n" + "="*40)
        for item in results:
            print(f"--- MATCH {item['rank']} ---")
            print(f"Source: {item.get('source')}")
            print(f"Distance: {item.get('distance', 'N/A')}")
            print(f"Text:\n{item.get('text')}\n")
