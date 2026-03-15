"""
Retrieval Augmented Generation (RAG) Pipeline
Coordinates context retrieval and prompt generation for an LLM to answer questions.
"""
import os
import sys
import argparse
from typing import Dict, Any

import logging
import time
from openai import OpenAI
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the system path to allow importing internal modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from retrieval.retrieve_context import ContextRetriever
from config.config import LLM_MODEL

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

class RAGPipeline:
    def __init__(self, index_name: str = "ai_knowledge"):
        self.retriever = ContextRetriever(index_name=index_name)
        
        # Verify API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY is not set.")
            
        self.client = OpenAI(api_key=api_key)
        self.llm_model = LLM_MODEL

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Retrieve relevant context from Endee
        2. Construct prompt
        3. Generate response using LLM
        """
        start_time = time.time()
        
        # Step 1: RetrieveContext
        retrieved_contexts = self.retriever.search(query=query, top_k=top_k)
        
        if not retrieved_contexts:
            duration = round(time.time() - start_time, 2)
            return {
                "answer": "I could not find any relevant context in the knowledge base to answer your question.",
                "sources": [],
                "contexts": [],
                "response_time_seconds": duration
            }
            
        # Step 2: Construct Prompt
        context_text = ""
        source_mapping = []
        snippets = []
        
        for idx, item in enumerate(retrieved_contexts, 1):
            source_file = item.get("source", "Unknown")
            text = item.get("text", "")
            distance = item.get("distance", 0.0)
            rank = item.get("rank", idx)
            
            # Format as numbered source for the prompt
            context_text += f"[{rank}] {source_file}\n{text}\n\n"
            
            source_mapping.append({
                "id": rank,
                "file": source_file
            })
            
            snippets.append({
                "rank": rank,
                "source": source_file,
                "text": text,
                "distance": distance
            })
            
        prompt = f"""You are an advanced AI Engineering Knowledge Engine assistant.
Use the provided context to answer the question. 
Cite sources using the bracket numbers like [1], [2] when referencing information.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context_text}

USER QUESTION: 
{query}

ANSWER:"""

        # Step 3: Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use provided context and cite using [number]."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response from LLM: {str(e)}"
            
        duration = round(time.time() - start_time, 2)
        
        # Observability Logging
        top_distance = retrieved_contexts[0].get("distance", 0.0) if retrieved_contexts else 0.0
        source_names = ", ".join(list(set([ctx["source"] for ctx in snippets])))
        
        logger.info(f"Query: {query}")
        logger.info(f"Sources: {source_names}")
        logger.info(f"Top Distance: {top_distance}")
        logger.info(f"Response Time: {duration} seconds")
            
        return {
            "answer": answer,
            "sources": source_mapping,
            "contexts": snippets,
            "response_time_seconds": duration
        }


if __name__ == "__main__":
    import json
    
    # Test through Command Line
    parser = argparse.ArgumentParser(description="AI Knowledge Engine RAG Pipeline")
    parser.add_argument("query", type=str, help="The question you want to ask the RAG system.")
    parser.add_argument("--k", type=int, default=3, help="Number of context chunks to retrieve.")
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    result = pipeline.answer_question(args.query, top_k=args.k)
    
    # Print exactly as structured JSON without extra logs
    print(json.dumps(result, indent=2))
