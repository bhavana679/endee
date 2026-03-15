"""
Embedding Generation and Storage Pipeline
Parses raw documents, chunks them, generates vectors (in batches), normalizes them,
and inserts them into Endee Vector Database avoiding duplicates.
"""

import os
import sys
import json
import uuid
import logging
import time

# Add the project root to the system path to allow importing internal modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from openai import OpenAI
from utils.text_processing import process_document
from config.config import DATA_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP, ENDEE_HOST, ENDEE_PORT
from retrieval.endee_client import EndeeClient

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, index_name: str = "ai_knowledge"):
        """
        Initialize the embedding pipeline.
        Connects to the OpenAI client and the Endee Vector Database.
        """
        logger.info(f"Using OpenAI embedding model '{EMBEDDING_MODEL_NAME}'")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = EMBEDDING_MODEL_NAME
        self.vector_dim = EMBEDDING_DIMENSION
        
        self.endee_client = EndeeClient(host=ENDEE_HOST, port=int(ENDEE_PORT))
        self.index_name = index_name
        logger.info("Initialization complete.")

    def _get_indexed_documents(self) -> set:
        # Load local state from cache file to avoid re-ingesting already processed files
        cache_file = os.path.join(BASE_DIR, ".ingested_docs.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _mark_document_indexed(self, filename: str):
        indexed = self._get_indexed_documents()
        indexed.add(filename)
        cache_file = os.path.join(BASE_DIR, ".ingested_docs.json")
        with open(cache_file, "w") as f:
            json.dump(list(indexed), f)

    def run_ingestion(self):
        """
        Main runner: creates the Endee index if it does not exist,
        processes all text documents, generates dense vectors (normalized, in batch),
        and saves them in the database.
        """
        # Tracking metrics for the ingestion run
        ingestion_stats = {
            "documents": 0,
            "chunks": 0,
            "vectors": 0,
            "duration_seconds": 0.0
        }
        
        start_time = time.time()

        # Ensure database is accessible
        try:
            health = self.endee_client.health()
            logger.info(f"Connected to Endee DB successful. Health timestamp: {health.get('timestamp')}")
        except Exception as e:
            logger.error(f"Failed to connect to Endee Vector Database at {ENDEE_HOST}:{ENDEE_PORT}.")
            logger.error("Please make sure the Endee server is running before executing this script.")
            sys.exit(1)

        # Check for and create the required index
        indexes = self.endee_client.list_indexes()
        if self.index_name not in indexes:
            logger.info(f"Index '{self.index_name}' not found. Creating it with dim={self.vector_dim} space_type='cosine'...")
            self.endee_client.create_index(index_name=self.index_name, dim=self.vector_dim, space_type="cosine")
        else:
            logger.info(f"Index '{self.index_name}' already exists.")

        # Iterate over all text documents in the data directory
        if not os.path.exists(DATA_DIR):
            logger.warning(f"Data directory '{DATA_DIR}' not found. Nothing to ingest.")
            return

        indexed_docs = self._get_indexed_documents()

        for filename in os.listdir(DATA_DIR):
            filepath = os.path.join(DATA_DIR, filename)
            
            # Skip subdirectories and non-text files
            if not os.path.isfile(filepath) or not filepath.endswith('.txt'):
                continue
            
            # Avoid Duplicate Ingestion
            if filename in indexed_docs:
                logger.info(f"Skipping '{filename}' - already indexed.")
                continue
                
            logger.info(f"Document processing started: '{filename}'")
            
            # Step 1: Extract and chunk the text
            chunks = process_document(filepath, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if not chunks:
                logger.warning(f"No valid text chunks found in '{filename}'.")
                continue
                
            logger.info(f"Number of chunks created: {len(chunks)}")
            ingestion_stats["chunks"] += len(chunks)

            # Step 2: Batch Embedding Generation via OpenAI
            logger.info(f"Generating batch embeddings for '{filename}'...")
            
            # Clean chunks for OpenAI (replace newlines)
            clean_chunks = [c.replace("\n", " ") for c in chunks]
            
            resp = self.client.embeddings.create(input=clean_chunks, model=self.model_name)
            embeddings = [data.embedding for data in resp.data]
            
            # Step 3: Prepare Endee payloads with metadata
            vectors_payload = []
            for chunk_id, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                doc_uuid = str(uuid.uuid4())
                
                metadata = {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text
                }

                vector_obj = {
                    "id": doc_uuid,
                    "vector": embedding.tolist(),
                    "meta": json.dumps(metadata)
                }
                vectors_payload.append(vector_obj)
            
            # Step 4: Insert vectors into the Endee instance
            if vectors_payload:
                success = self.endee_client.insert_vectors(
                    index_name=self.index_name,
                    vectors=vectors_payload
                )
                if success:
                    inserted_len = len(vectors_payload)
                    logger.info(f"Vectors inserted: {inserted_len} chunks for '{filename}'.")
                    self._mark_document_indexed(filename)
                    
                    ingestion_stats["documents"] += 1
                    ingestion_stats["vectors"] += inserted_len
                else:
                    logger.error(f"Failed to ingest vectors for '{filename}'. Please check Endee logs.")

        end_time = time.time()
        ingestion_stats["duration_seconds"] = round(end_time - start_time, 2)
        
        logger.info("Ingestion completed.")
        
        print("\n" + "="*40)
        print("## Ingestion Summary")
        print("="*40)
        print(f"Documents processed: {ingestion_stats['documents']}")
        print(f"Chunks created:      {ingestion_stats['chunks']}")
        print(f"Vectors stored:      {ingestion_stats['vectors']}")
        print(f"Total runtime:       {ingestion_stats['duration_seconds']} seconds")
        print("="*40 + "\n")


if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    pipeline.run_ingestion()
