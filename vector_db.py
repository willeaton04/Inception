#!/usr/bin/env python3
"""
Provides semantic search and file embedding capabilities for AgenticFileScraper
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import os

# Disable ChromaDB telemetry before importing
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ChromaDB and embedding dependencies
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("Please install required packages:")
    print("pip install chromadb sentence-transformers")
    raise e


class VectorDatabase:
    """ChromaDB-based vector database for semantic file search"""

    def __init__(self, db_path: str = "file_vectors.db", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB vector database

        Args:
            db_path: Path for ChromaDB persistence
            model_name: Sentence transformer model name
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize ChromaDB with persistence (disable telemetry to avoid errors)
        self.logger.info(f"Initializing ChromaDB at {self.db_path}")

        # Disable telemetry completely to avoid errors
        import chromadb.config
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        # Set environment variable to disable telemetry
        import os
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=settings
        )

        # Create or get collection for file embeddings
        try:
            self.collection = self.client.get_collection(
                name="file_embeddings"
            )
            self.logger.info(f"Loaded existing collection with {self.collection.count()} chunks")
        except:
            self.collection = self.client.create_collection(
                name="file_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("Created new collection")

        # Track file chunks for efficient management
        self._file_chunk_index = self._rebuild_file_index()

    def _rebuild_file_index(self) -> Dict[str, List[str]]:
        """Rebuild the file-to-chunk index from ChromaDB"""
        file_index = {}

        try:
            # Get all documents to build index
            results = self.collection.get()

            if results and results['metadatas']:
                for chunk_id, metadata in zip(results['ids'], results['metadatas']):
                    file_path = metadata.get('file_path', '')
                    if file_path:
                        if file_path not in file_index:
                            file_index[file_path] = []
                        file_index[file_path].append(chunk_id)

            self.logger.info(f"Rebuilt index for {len(file_index)} files")
        except Exception as e:
            self.logger.error(f"Error rebuilding file index: {str(e)}")

        return file_index

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """Generate unique ID for a chunk"""
        return hashlib.md5(f"{file_path}_chunk_{chunk_index}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:  # Only break if we have decent chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap if end < text_len else text_len

        return [c for c in chunks if c]  # Filter empty chunks

    def store_file(self, file_path: Path, content: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Store file content in ChromaDB

        Args:
            file_path: Path to the file
            content: File content
            metadata: Additional metadata

        Returns:
            File ID if successful, None otherwise
        """
        try:
            file_path_str = str(file_path)

            # Delete existing chunks for this file (if any)
            if file_path_str in self._file_chunk_index:
                self.delete_file(file_path_str)

            # Generate file ID
            file_id = hashlib.md5(f"{file_path_str}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

            # Chunk the content
            chunks = self._chunk_text(content)

            if not chunks:
                self.logger.warning(f"No chunks generated for {file_path_str}")
                return None

            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(file_path_str, i)

                # Create metadata for this chunk
                chunk_metadata = {
                    'file_path': file_path_str,
                    'file_id': file_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'timestamp': datetime.now().isoformat()
                }

                # Add custom metadata
                if metadata:
                    # Convert non-string values to strings for ChromaDB
                    for key, value in metadata.items():
                        if not isinstance(value, (str, int, float, bool)):
                            chunk_metadata[key] = str(value)
                        else:
                            chunk_metadata[key] = value

                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadatas.append(chunk_metadata)

            # Generate embeddings using sentence-transformers
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=False).tolist()

            # Store in ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )

            # Update file index
            self._file_chunk_index[file_path_str] = chunk_ids

            self.logger.info(f"Stored {len(chunks)} chunks for {file_path.name}")
            return file_id

        except Exception as e:
            self.logger.error(f"Error storing file {file_path}: {str(e)}")
            return None

    def search_similar(self, query_text: str, limit: int = 10, threshold: float = 0.0,
                       file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content using ChromaDB's semantic search

        Args:
            query_text: Query string
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            file_filter: Optional file path to filter results

        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False).tolist()

            # Build where clause for filtering
            where_clause = None
            if file_filter:
                where_clause = {"file_path": {"$eq": file_filter}}

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results for filtering
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )

            if not results or not results['ids'] or not results['ids'][0]:
                return []

            # Process results
            search_results = []
            seen_files = set()

            for idx in range(len(results['ids'][0])):
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - results['distances'][0][idx]

                # Apply threshold
                if similarity < threshold:
                    continue

                metadata = results['metadatas'][0][idx]
                document = results['documents'][0][idx]

                # Optional: deduplicate by file
                file_path = metadata.get('file_path', '')

                search_results.append({
                    'file_path': file_path,
                    'content': document,
                    'similarity': float(similarity),
                    'metadata': metadata,
                    'chunk_index': metadata.get('chunk_index', 0)
                })

                seen_files.add(file_path)

                # Stop if we have enough results
                if len(search_results) >= limit:
                    break

            return search_results

        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []

    def get_file_chunks(self, file_path: str) -> List[str]:
        """Get all text chunks for a specific file"""
        try:
            if file_path not in self._file_chunk_index:
                return []

            chunk_ids = self._file_chunk_index[file_path]

            # Get chunks from ChromaDB
            results = self.collection.get(
                ids=chunk_ids,
                include=["documents"]
            )

            if results and results['documents']:
                # Sort by chunk index
                chunks_with_index = []
                for i, doc in enumerate(results['documents']):
                    # Get the metadata to find chunk index
                    metadata_results = self.collection.get(ids=[chunk_ids[i]], include=["metadatas"])
                    chunk_index = metadata_results['metadatas'][0].get('chunk_index', i)
                    chunks_with_index.append((chunk_index, doc))

                # Sort and return just the documents
                chunks_with_index.sort(key=lambda x: x[0])
                return [doc for _, doc in chunks_with_index]

            return []

        except Exception as e:
            self.logger.error(f"Error getting chunks for {file_path}: {str(e)}")
            return []

    def delete_file(self, file_path: str) -> bool:
        """
        Remove all embeddings for a file from ChromaDB

        Args:
            file_path: Path to file to remove

        Returns:
            True if successful
        """
        try:
            if file_path not in self._file_chunk_index:
                return False

            # Get chunk IDs for this file
            chunk_ids = self._file_chunk_index[file_path]

            # Delete from ChromaDB
            self.collection.delete(ids=chunk_ids)

            # Update index
            del self._file_chunk_index[file_path]

            self.logger.info(f"Deleted {len(chunk_ids)} chunks for {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""

        try:
            total_chunks = self.collection.count()
            total_files = len(self._file_chunk_index)

            # Calculate average chunks per file
            avg_chunks = total_chunks / total_files if total_files > 0 else 0

            # Get metadata sample to analyze file types
            sample_results = self.collection.get(limit=min(100, total_chunks)) if total_chunks > 0 else None
            file_types = {}

            if sample_results and sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1

            # Get model name safely
            try:
                # Try to get model name from the model's modules
                model_name = self.embedding_model.modules()[0].auto_model.__class__.__name__ if hasattr(
                    self.embedding_model, 'modules') else 'all-MiniLM-L6-v2'
            except:
                model_name = 'all-MiniLM-L6-v2'  # Default fallback

            return {
                'total_files': total_files,
                'total_chunks': total_chunks,
                'avg_chunks_per_file': round(avg_chunks, 2),
                'embedding_dimension': self.embedding_dim,
                'embedding_model': model_name,
                'file_types': file_types,
                'database_path': str(self.db_path),
                'database_type': 'ChromaDB'
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {
                'error': str(e),
                'database_path': str(self.db_path),
                'database_type': 'ChromaDB'
            }

    def reset_database(self):
        """
        Completely reset the database (useful for fixing corruption)
        Warning: This will delete all stored data!
        """
        try:
            self.logger.warning("Resetting database - all data will be lost!")

            # Delete the collection
            try:
                self.client.delete_collection(name="file_embeddings")
            except:
                pass  # Collection might not exist

            # Recreate collection
            self.collection = self.client.create_collection(
                name="file_embeddings",
                metadata={"hnsw:space": "cosine"}
            )

            # Clear the file index
            self._file_chunk_index = {}

            self.logger.info("Database reset complete")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting database: {str(e)}")
            return False

    def clear_database(self):
        """Clear all data from the database (alias for reset_database)"""
        return self.reset_database()

    def find_duplicates(self, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Find potentially duplicate files based on embedding similarity

        Args:
            threshold: Similarity threshold for considering files as duplicates

        Returns:
            List of (file1, file2, similarity) tuples
        """
        duplicates = []

        try:
            # Get representative chunks for each file
            file_embeddings = {}

            for file_path, chunk_ids in self._file_chunk_index.items():
                if chunk_ids:
                    # Get first chunk as representative
                    results = self.collection.get(
                        ids=[chunk_ids[0]],
                        include=["embeddings"]
                    )

                    if results and results['embeddings']:
                        file_embeddings[file_path] = np.array(results['embeddings'][0])

            # Compare all pairs
            file_paths = list(file_embeddings.keys())
            for i in range(len(file_paths)):
                for j in range(i + 1, len(file_paths)):
                    # Cosine similarity
                    similarity = np.dot(file_embeddings[file_paths[i]],
                                        file_embeddings[file_paths[j]])

                    if similarity >= threshold:
                        duplicates.append((file_paths[i], file_paths[j], float(similarity)))

        except Exception as e:
            self.logger.error(f"Error finding duplicates: {str(e)}")

        return duplicates

    def get_related_files(self, file_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find files related to a given file

        Args:
            file_path: Path to the reference file
            limit: Maximum number of related files to return

        Returns:
            List of related files with similarity scores
        """
        try:
            if file_path not in self._file_chunk_index:
                return []

            # Get chunks for this file
            chunk_ids = self._file_chunk_index[file_path]
            if not chunk_ids:
                return []

            # Get the first chunk's content as query
            results = self.collection.get(
                ids=[chunk_ids[0]],
                include=["documents"]
            )

            if not results or not results['documents']:
                return []

            query_text = results['documents'][0]

            # Search for similar content, excluding the source file
            search_results = self.search_similar(query_text, limit=limit + 10)

            # Filter out the source file and aggregate by file
            file_scores = {}
            for result in search_results:
                result_path = result['file_path']
                if result_path != file_path:
                    if result_path not in file_scores:
                        file_scores[result_path] = []
                    file_scores[result_path].append(result['similarity'])

            # Average scores per file
            related_files = []
            for path, scores in file_scores.items():
                avg_similarity = sum(scores) / len(scores)
                related_files.append({
                    'file_path': path,
                    'similarity': float(avg_similarity),
                    'chunk_matches': len(scores)
                })

            # Sort by similarity and limit
            related_files.sort(key=lambda x: x['similarity'], reverse=True)
            return related_files[:limit]

        except Exception as e:
            self.logger.error(f"Error finding related files: {str(e)}")
            return []

    def batch_store_files(self, file_contents: List[Tuple[Path, str, Dict[str, Any]]]) -> List[Optional[str]]:
        """
        Store multiple files efficiently in batch

        Args:
            file_contents: List of (file_path, content, metadata) tuples

        Returns:
            List of file IDs (None for failed files)
        """
        file_ids = []

        all_chunks = []
        all_ids = []
        all_metadatas = []
        file_id_map = {}

        try:
            for file_path, content, metadata in file_contents:
                file_path_str = str(file_path)

                # Delete existing chunks
                if file_path_str in self._file_chunk_index:
                    self.delete_file(file_path_str)

                # Generate file ID
                file_id = hashlib.md5(f"{file_path_str}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
                file_ids.append(file_id)

                # Chunk the content
                chunks = self._chunk_text(content)

                if not chunks:
                    file_ids[-1] = None
                    continue

                chunk_ids_for_file = []

                for i, chunk in enumerate(chunks):
                    chunk_id = self._generate_chunk_id(file_path_str, i)

                    chunk_metadata = {
                        'file_path': file_path_str,
                        'file_id': file_id,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                        'timestamp': datetime.now().isoformat()
                    }

                    if metadata:
                        for key, value in metadata.items():
                            if not isinstance(value, (str, int, float, bool)):
                                chunk_metadata[key] = str(value)
                            else:
                                chunk_metadata[key] = value

                    all_ids.append(chunk_id)
                    all_chunks.append(chunk)
                    all_metadatas.append(chunk_metadata)
                    chunk_ids_for_file.append(chunk_id)

                file_id_map[file_path_str] = chunk_ids_for_file

            # Batch generate embeddings
            if all_chunks:
                self.logger.info(f"Generating embeddings for {len(all_chunks)} total chunks")
                embeddings = self.embedding_model.encode(all_chunks, convert_to_tensor=False).tolist()

                # Batch store in ChromaDB
                self.collection.add(
                    ids=all_ids,
                    embeddings=embeddings,
                    documents=all_chunks,
                    metadatas=all_metadatas
                )

                # Update file index
                self._file_chunk_index.update(file_id_map)

                self.logger.info(f"Batch stored {len(file_contents)} files")

        except Exception as e:
            self.logger.error(f"Error in batch store: {str(e)}")

        return file_ids


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create database
    db = VectorDatabase("db/chromadb_vectors")

    # Test storing a file
    test_content = """
    This is a test document about Python programming.
    Python is a high-level programming language.
    It is known for its simplicity and readability.
    Many developers use Python for web development, data science, and automation.
    """

    file_id = db.store_file(
        Path("webscrapper/test.py"),
        test_content,
        {"file_type": "text/python", "project": "test"}
    )

    print(f"Stored file with ID: {file_id}")

    # Test searching
    results = db.search_similar("Python programming language", limit=5)
    print(f"\nSearch results: {len(results)} found")
    for result in results:
        print(f"  - {result['file_path']}: {result['similarity']:.3f}")
        print(f"    Content: {result['content'][:100]}...")

    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(json.dumps(stats, indent=2))