#!/usr/bin/env python3
"""
Vector database implementation using SQLite and sentence transformers
"""

import sqlite3
import hashlib
import pickle
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print('\033[1;31m[Error]:\033[0m sentence-transformers not found. Install with: pip install sentence-transformers')
    raise


class VectorDatabase:
    """Local vector database using SQLite and sentence transformers"""

    def __init__(self, db_path: str = "file_vectors.db", model_name: str = 'all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

        # Initialize database first
        self.init_database()

        # Then load the embedding model
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            print(f'\033[1;33m[Vector DB]:\033[0m Loading embedding model {self.model_name}...')
            self.model = SentenceTransformer(self.model_name)

            # Get embedding dimension
            test_embedding = self.model.encode(['test'])
            self.embedding_dim = len(test_embedding[0])

            print(f'\033[1;32m[Vector DB]:\033[0m Model loaded (dim: {self.embedding_dim})')

        except Exception as e:
            print(f'\033[1;31m[Vector DB Error]:\033[0m Failed to load model: {str(e)}')
            raise

    def init_database(self):
        """Initialize SQLite database for vectors"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # File embeddings table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS file_embeddings
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           file_path
                           TEXT,
                           file_hash
                           TEXT,
                           content_chunk
                           TEXT,
                           chunk_index
                           INTEGER,
                           embedding
                           BLOB,
                           metadata
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           updated_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           UNIQUE
                       (
                           file_path,
                           chunk_index
                       )
                           )
                       ''')

        # Scan sessions table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS scan_sessions
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           session_id
                           TEXT
                           UNIQUE,
                           goal
                           TEXT,
                           file_count
                           INTEGER,
                           findings
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON file_embeddings(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON file_embeddings(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON scan_sessions(session_id)')

        conn.commit()
        conn.close()

        print(f'\033[1;34m[Vector DB]:\033[0m Database initialized at {self.db_path}')

    def get_file_hash(self, file_path: Path) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with overlap for better context"""
        if len(text) <= max_chunk_size:
            return [text]

        # Try to split by sentences first
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap if possible
                    if overlap > 0 and len(current_chunk) > overlap:
                        # Take last 'overlap' characters as starting point for next chunk
                        current_chunk = current_chunk[-overlap:] + " " + sentence + ". "
                    else:
                        current_chunk = sentence + ". "
                else:
                    # Sentence itself is too long, split it
                    if len(sentence) > max_chunk_size:
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                                temp_chunk += word + " "
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word + " "
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:max_chunk_size]]

    def embed_file(self, file_path: Path, content: str, metadata: Dict = None) -> bool:
        """Create embeddings for file content"""
        if not self.model:
            print('\033[1;31m[Vector DB Error]:\033[0m Embedding model not loaded')
            return False

        try:
            file_hash = self.get_file_hash(file_path)

            # Check if file already embedded and unchanged
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT file_hash FROM file_embeddings WHERE file_path = ? LIMIT 1',
                (str(file_path),)
            )
            result = cursor.fetchone()

            if result and result[0] == file_hash:
                conn.close()
                return True  # Already embedded and unchanged

            # Delete old embeddings for this file
            cursor.execute('DELETE FROM file_embeddings WHERE file_path = ?', (str(file_path),))

            # Chunk content and create embeddings
            chunks = self.chunk_text(content)

            if not chunks:
                conn.close()
                return False

            # Create embeddings for all chunks at once (more efficient)
            embeddings = self.model.encode(chunks)

            # Store each chunk with its embedding
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if len(chunk.strip()) < 30:  # Skip very small chunks
                    continue

                embedding_blob = pickle.dumps(embedding)

                # Store in database
                cursor.execute('''
                    INSERT OR REPLACE INTO file_embeddings 
                    (file_path, file_hash, content_chunk, chunk_index, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    str(file_path),
                    file_hash,
                    chunk,
                    i,
                    embedding_blob,
                    json.dumps(metadata or {})
                ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Failed to embed {file_path.name}: {str(e)}')
            return False

    def semantic_search(self, query: str, top_k: int = 10, file_filter: List[str] = None,
                        similarity_threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """Search for semantically similar content"""
        if not self.model:
            print('\033[1;31m[Vector DB Error]:\033[0m Embedding model not loaded')
            return []

        try:
            query_embedding = self.model.encode([query])[0]

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build query with optional file filter
            where_clause = ""
            params = []
            if file_filter:
                placeholders = ','.join(['?' for _ in file_filter])
                where_clause = f"WHERE file_path IN ({placeholders})"
                params = file_filter

            cursor.execute(f'''
                SELECT file_path, content_chunk, embedding, metadata 
                FROM file_embeddings 
                {where_clause}
            ''', params)

            results = []
            for file_path, content_chunk, embedding_blob, metadata_json in cursor.fetchall():
                try:
                    stored_embedding = pickle.loads(embedding_blob)

                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )

                    # Only include results above threshold
                    if similarity >= similarity_threshold:
                        results.append((file_path, content_chunk, float(similarity)))

                except Exception as e:
                    continue  # Skip corrupted embeddings

            conn.close()

            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Search failed: {str(e)}')
            return []

    def get_similar_files(self, file_path: Path, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find files similar to the given file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get embeddings for the target file
            cursor.execute('''
                           SELECT embedding
                           FROM file_embeddings
                           WHERE file_path = ?
                           ''', (str(file_path),))

            target_embeddings = []
            for (embedding_blob,) in cursor.fetchall():
                try:
                    embedding = pickle.loads(embedding_blob)
                    target_embeddings.append(embedding)
                except:
                    continue

            if not target_embeddings:
                conn.close()
                return []

            # Average the embeddings for the target file
            target_avg = np.mean(target_embeddings, axis=0)

            # Get embeddings for all other files
            cursor.execute('''
                           SELECT file_path, embedding
                           FROM file_embeddings
                           WHERE file_path != ?
                           ''', (str(file_path),))

            file_similarities = {}
            for other_path, embedding_blob in cursor.fetchall():
                try:
                    other_embedding = pickle.loads(embedding_blob)
                    similarity = np.dot(target_avg, other_embedding) / (
                            np.linalg.norm(target_avg) * np.linalg.norm(other_embedding)
                    )

                    # Keep track of highest similarity per file
                    if other_path not in file_similarities or similarity > file_similarities[other_path]:
                        file_similarities[other_path] = float(similarity)

                except:
                    continue

            conn.close()

            # Sort and return top similar files
            sorted_files = sorted(file_similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_files[:top_k]

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Similar files search failed: {str(e)}')
            return []

    def store_scan_session(self, session_id: str, goal: str, file_count: int, findings: str) -> bool:
        """Store scan session results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scan_sessions 
                (session_id, goal, file_count, findings)
                VALUES (?, ?, ?, ?)
            ''', (session_id, goal, file_count, findings))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Failed to store session: {str(e)}')
            return False

    def get_scan_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scan sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT session_id, goal, file_count, findings, created_at
                           FROM scan_sessions
                           ORDER BY created_at DESC LIMIT ?
                           ''', (limit,))

            sessions = []
            for session_id, goal, file_count, findings, created_at in cursor.fetchall():
                sessions.append({
                    'session_id': session_id,
                    'goal': goal,
                    'file_count': file_count,
                    'findings': findings,
                    'created_at': created_at
                })

            conn.close()
            return sessions

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Failed to get history: {str(e)}')
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count total embeddings
            cursor.execute('SELECT COUNT(*) FROM file_embeddings')
            total_embeddings = cursor.fetchone()[0]

            # Count unique files
            cursor.execute('SELECT COUNT(DISTINCT file_path) FROM file_embeddings')
            unique_files = cursor.fetchone()[0]

            # Count scan sessions
            cursor.execute('SELECT COUNT(*) FROM scan_sessions')
            total_sessions = cursor.fetchone()[0]

            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]

            conn.close()

            return {
                'total_embeddings': total_embeddings,
                'unique_files': unique_files,
                'total_sessions': total_sessions,
                'database_size_bytes': db_size,
                'model_name': self.model_name,
                'embedding_dimension': self.embedding_dim
            }

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Failed to get stats: {str(e)}')
            return {}

    def cleanup_old_embeddings(self, days: int = 30) -> int:
        """Remove embeddings older than specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                DELETE FROM file_embeddings 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            print(f'\033[1;32m[Vector DB]:\033[0m Cleaned up {deleted_count} old embeddings')
            return deleted_count

        except Exception as e:
            print(f'\033[1;33m[Vector DB Warning]:\033[0m Cleanup failed: {str(e)}')
            return 0