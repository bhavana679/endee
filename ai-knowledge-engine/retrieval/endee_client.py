import os
import requests
import json
import msgpack

class EndeeClient:
    """
    A minimal API client for interacting with the local Endee Vector Database.
    """
    def __init__(self, host: str = "localhost", port: int = 8080, auth_token: str = None):
        # Determine protocol based on port (443 is usually HTTPS)
        protocol = "https" if str(port) == "443" else "http"
        self.base_url = f"{protocol}://{host}:{port}/api/v1"
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = auth_token

    def health(self) -> dict:
        """Check the health of the Endee system."""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_indexes(self) -> list[str]:
        """List all available indexes in the vector database."""
        response = requests.get(f"{self.base_url}/index/list", headers=self.headers)
        if response.status_code == 200:
            indexes = response.json().get("indexes", [])
            return [idx.get("name") for idx in indexes]
        return []

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine") -> bool:
        """
        Create a new dense vector index in Endee.
        Returns True if successful, False otherwise.
        """
        url = f"{self.base_url}/index/create"
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type
        }
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 409:
            # Index already exists
            return True
            
        print(f"Error creating index: {response.text}")
        return False

    def insert_vectors(self, index_name: str, vectors: list[dict]) -> bool:
        """
        Insert an array of vector records into the exact index.
        The records should be dicts containing:
        - "id": string or int
        - "vector": array of floats
        - "meta": serialized string of dict containing metadata
        """
        url = f"{self.base_url}/index/{index_name}/vector/insert"
        response = requests.post(url, headers=self.headers, json=vectors)
        if response.status_code == 200:
            return True
            
        print(f"Error inserting vectors: {response.text}")
        return False

    def search(self, index_name: str, query_vector: list[float], k: int = 3) -> list[dict]:
        """
        Search for the top K most similar vectors in Endee.
        Expects a MessagePack binary response from the Endee server.
        """
        url = f"{self.base_url}/index/{index_name}/search"
        payload = {
            "vector": query_vector,
            "k": k,
            "include_vectors": False
        }
        
        # Endee returns msgpack format for dense search results
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error querying Endee: {response.status_code} - {response.text}")
            return []
            
        try:
            # We must use raw response.content because it is msgpack binary
            unpacked_resp = msgpack.unpackb(response.content, raw=False)
            
            # Endee's format for ResultSet usually looks like:
            # {"data": [{"id": "...", "distance": 0.89, "meta": "{...}"}, ...]}
            # For simplicity, we assume it's a list or has a "data" key.
            if isinstance(unpacked_resp, dict) and "data" in unpacked_resp:
                return unpacked_resp["data"]
            elif isinstance(unpacked_resp, list):
                return unpacked_resp
            return unpacked_resp
            
        except msgpack.exceptions.ExtraData as e:
            print(f"Failed to unpack msgpack response: {e}")
            return []

