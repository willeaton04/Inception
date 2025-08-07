# AgenticFileScraper and VectorDatabase Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [AgenticFileScraper Deep Dive](#agenticfilescraper-deep-dive)
4. [VectorDatabase Deep Dive](#vectordatabase-deep-dive)
5. [Data Flow](#data-flow)
6. [AI Integration](#ai-integration)
7. [Performance Optimizations](#performance-optimizations)
8. [Use Cases and Examples](#use-cases-and-examples)

## System Overview

The AgenticFileScraper is an intelligent file analysis system that combines AI-powered content understanding with vector-based semantic search. It's designed to automatically scan, analyze, and extract insights from codebases and document repositories based on natural language goals.

### Key Capabilities
- **Goal-driven analysis**: Converts natural language objectives into actionable scanning strategies
- **Semantic understanding**: Uses vector embeddings to find conceptually related content
- **AI-powered insights**: Leverages LLMs for deep content analysis and pattern recognition
- **Adaptive prioritization**: Intelligently prioritizes files based on relevance signals
- **Context preservation**: Maintains analysis context across files for holistic insights

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input (Goal)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgenticFileScraper                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Goal Parser (AI-powered)                │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │            ScanStrategy Generation                   │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │         File Discovery & Prioritization              │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │              File Analysis Pipeline                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │    │
│  │  │  Reader  │→ │    AI    │→ │ Vector Storage   │  │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │           Semantic Analysis & Insights               │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │         Summary & Recommendations (AI)               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    VectorDatabase                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Embedding Generation                    │    │
│  │                 (Sentence Transformers)              │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐    │
│  │              ChromaDB Vector Store                   │    │
│  │         (Persistent Storage + HNSW Index)            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## AgenticFileScraper Deep Dive

### 1. Initialization and Setup

```python
def __init__(self, goal: str, ollama_model: str = 'llama3', vector_db_path: str = "file_vectors.db")
```

**Under the hood:**
- **Session ID Generation**: Creates a unique MD5 hash from the goal and timestamp for tracking
- **Component Initialization**: 
  - Sets up OllamaManager for LLM interactions
  - Initializes VectorDatabase with persistent storage
  - Creates error tracking structures
- **Goal Parsing**: Immediately converts the natural language goal into a structured strategy

### 2. Goal Parsing and Strategy Generation

The `_parse_goal()` method is the brain of the system's planning phase:

**Process:**
1. **Prompt Engineering**: Constructs a specialized system prompt that instructs the LLM to act as a "file analysis strategist"
2. **LLM Analysis**: Sends the goal to the LLM with strict JSON output requirements
3. **Response Parsing**: Extracts and validates JSON from LLM response
4. **Strategy Creation**: Builds a `ScanStrategy` object with:
   - **Keywords**: Extracted search terms relevant to the goal
   - **File priorities**: Extensions to focus on (e.g., `.py` for Python analysis)
   - **Analysis depth**: Shallow/medium/deep scanning intensity
   - **Patterns to find**: Specific code patterns or anti-patterns
   - **Scan focus**: Category (security/bugs/performance/documentation/structure/general)
   - **Semantic queries**: Natural language queries for vector search

**Fallback Mechanism**: If JSON parsing fails, creates a default strategy based on common patterns.

### 3. File Discovery and Intelligent Prioritization

The file discovery system uses a multi-stage filtering and scoring approach:

#### Discovery Phase (`find_files()`):
```python
def find_files(self, path: Path, extensions: List[str] = None) -> List[Path]
```

**Filtering Pipeline:**
1. **Hidden File Exclusion**: Skips files/directories starting with '.'
2. **Build Artifact Filtering**: Ignores `node_modules`, `__pycache__`, `.git`, `build`, `dist`, etc.
3. **Size Limiting**: Rejects files exceeding strategy-defined limits (default ~10MB)
4. **Extension Filtering**: Optionally filters by file extensions

#### Prioritization Algorithm (`_prioritize_files()`):

Each file receives a priority score based on multiple factors:

```python
def priority_score(file_path: Path) -> float:
    score = 0.0
    
    # High-priority file detection (+15 points)
    # - Main entry points (main.py, index.js, app.py)
    # - Configuration files (config.*, settings.*)
    # - Security-related files (auth.*, security.*)
    
    # Extension matching (+10 points)
    # - Matches against strategy.file_priorities
    
    # Keyword presence in path (+5 points per keyword)
    # - Scans full path for strategy keywords
    
    # Focus-specific keywords (+3 points per match)
    # - Security focus: auth, login, password, secret, key, token
    # - Performance focus: cache, optimize, performance, speed
    # - Etc.
    
    # Size-based scoring:
    # - Sweet spot (1KB-50KB): +4 points
    # - Medium files (50KB-200KB): +2 points
    # - Large files (200KB-500KB): +1 point
    
    return score
```

Files are then sorted by descending score, ensuring the most relevant files are analyzed first.

### 4. Individual File Analysis

The `analyze_file()` method orchestrates deep analysis of each file:

#### Content Reading (`_read_file_safely()`):
- **Multi-encoding support**: Attempts UTF-8, Latin-1, CP1252, ISO-8859-1, UTF-16
- **Validation**: Skips nearly empty files (<10 characters)
- **Error resilience**: Gracefully handles encoding failures

#### Vector Storage:
```python
metadata = {
    'file_type': file_type,        # MIME type
    'size': file_stats.st_size,    # File size in bytes
    'modified': file_stats.st_mtime, # Last modification time
    'scan_session': self.session_id, # Current session ID
    'goal': self.goal              # Original scanning goal
}

file_id = self.vector_db.store_file(file_path, content, metadata)
```

#### Semantic Matching:
For each semantic query in the strategy:
1. Searches for similar content chunks within the current file
2. Retrieves top 3 matches above 0.6 similarity threshold
3. Stores matches with query context for AI analysis

#### AI Analysis (`_ai_analyze_file()`):

**Content Preparation:**
- Truncates content based on analysis depth (1000-5000 chars)
- Builds semantic context from vector matches
- Constructs focused prompts for the LLM

**LLM Processing:**
```python
# System prompt defines expected JSON structure
{
    "relevance_score": 0.0-1.0,
    "key_findings": ["finding1", "finding2"],
    "content_summary": "Brief description",
    "recommendations": ["action1", "action2"]
}
```

**JSON Parsing Strategies:**
1. **Direct parsing**: Attempts standard JSON parsing
2. **Regex extraction**: Falls back to pattern matching for malformed JSON
3. **Enhanced fallback**: Uses heuristic analysis if AI fails

### 5. Semantic Analysis Across Files

The `run_semantic_analysis()` method performs cross-file semantic search:

```python
for query in self.strategy.semantic_queries:
    search_results = self.vector_db.search_similar(
        query_text=query,
        limit=15,
        threshold=0.5
    )
```

**Process:**
1. Executes each semantic query against the entire vector database
2. Retrieves top 15 matches across all files
3. Creates `SemanticInsight` objects with confidence scoring
4. Identifies high-confidence patterns (similarity > 0.7)

### 6. Summary and Recommendation Generation

#### Summary Generation (`_generate_summary()`):

**Data Aggregation:**
- Calculates relevance statistics (average, high-relevance count)
- Extracts top findings from highest-scoring files
- Computes file type distribution
- Identifies semantic patterns

**AI Summary Creation:**
- Provides comprehensive context to LLM
- Requests 2-3 sentence technical summary
- Falls back to statistical summary if AI fails

#### Recommendation Engine (`_generate_recommendations()`):

**Multi-source Recommendations:**
1. **File-level**: Aggregates recommendations from individual file analyses
2. **Strategy-based**: Adds focus-specific recommendations (security, performance, etc.)
3. **Semantic-based**: Suggests actions based on pattern detection
4. **Deduplication**: Removes redundant recommendations
5. **Limiting**: Returns top 8 most actionable items

## VectorDatabase Deep Dive

### 1. Core Components

**Embedding Model:**
```python
self.model = SentenceTransformer('all-MiniLM-L6-v2')
```
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Optimization**: Balanced between quality and speed
- **Use case**: Optimized for semantic similarity tasks

**Vector Store (ChromaDB):**
```python
self.client = chromadb.PersistentClient(path=str(self.db_path))
self.collection = self.client.get_or_create_collection(
    name="file_embeddings",
    metadata={"hnsw:space": "cosine"}
)
```
- **Storage**: Persistent SQLite backend
- **Index**: HNSW (Hierarchical Navigable Small World) graphs
- **Distance metric**: Cosine similarity
- **Advantages**: Fast approximate nearest neighbor search

### 2. Document Processing Pipeline

#### Chunking Strategy:
```python
def _chunk_content(self, content: str, chunk_size: int = 500, overlap: int = 50)
```

**Process:**
1. **Sentence Splitting**: Uses NLTK or regex to identify sentence boundaries
2. **Chunk Assembly**: Groups sentences to reach target chunk size
3. **Overlap Management**: Includes context from previous chunk
4. **Size Optimization**: Ensures chunks are neither too small nor too large

**Why Chunking?**
- **Context preservation**: Maintains semantic coherence
- **Embedding quality**: Optimal input size for transformer models
- **Search granularity**: Enables precise content matching
- **Memory efficiency**: Manages large documents effectively

### 3. Embedding Generation and Storage

```python
def store_file(self, file_path: Path, content: str, metadata: dict)
```

**Embedding Process:**
1. **Content chunking**: Splits content into overlapping segments
2. **Batch encoding**: Generates embeddings for all chunks simultaneously
3. **ID generation**: Creates unique identifiers for each chunk
4. **Metadata enrichment**: Adds file path, chunk index, timestamps

**Storage Structure:**
```python
{
    "ids": ["file_hash_chunk_0", "file_hash_chunk_1"],
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "documents": ["chunk content 1", "chunk content 2"],
    "metadatas": [{
        "file_path": "/path/to/file",
        "chunk_index": 0,
        "total_chunks": 10,
        "file_type": "text/python",
        "timestamp": "2024-01-01T00:00:00"
    }]
}
```

### 4. Semantic Search Implementation

```python
def search_similar(self, query_text: str, limit: int = 10, threshold: float = 0.0)
```

**Search Pipeline:**

1. **Query Embedding**:
   ```python
   query_embedding = self.model.encode(query_text, convert_to_tensor=False)
   ```

2. **Vector Search**:
   - ChromaDB performs HNSW search
   - Returns K nearest neighbors
   - Calculates cosine similarity scores

3. **Result Processing**:
   - Filters by similarity threshold
   - Deduplicates overlapping chunks
   - Enriches with metadata
   - Sorts by relevance

**Optimization Techniques:**
- **Index tuning**: HNSW parameters optimized for recall/speed trade-off
- **Batch querying**: Multiple queries processed together when possible
- **Caching**: Frequently accessed embeddings cached in memory

### 5. Deduplication and Result Merging

**Challenge**: Multiple chunks from the same file may match a query

**Solution**:
```python
def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
    seen_files = {}
    deduplicated = []
    
    for result in results:
        file_path = result['file_path']
        if file_path not in seen_files:
            seen_files[file_path] = result
        else:
            # Keep the highest scoring chunk
            if result['similarity'] > seen_files[file_path]['similarity']:
                seen_files[file_path] = result
    
    return list(seen_files.values())
```

## Data Flow

### Complete Analysis Flow:

1. **User Input**: "Find security vulnerabilities in authentication code"

2. **Strategy Generation**:
   ```json
   {
     "keywords": ["auth", "login", "password", "security"],
     "file_priorities": [".py", ".js", ".java"],
     "scan_focus": "security",
     "semantic_queries": ["authentication vulnerabilities", "password handling"]
   }
   ```

3. **File Discovery**:
   - Scans directory tree
   - Filters 1000 files down to 200 relevant ones
   - Prioritizes based on scoring algorithm
   - Selects top 50 for analysis

4. **Per-File Processing**:
   ```
   File → Read → Chunk → Embed → Store → AI Analyze → Results
   ```

5. **Semantic Analysis**:
   ```
   All Embeddings → Query Search → Pattern Detection → Insights
   ```

6. **Final Output**:
   - Ranked list of high-relevance files
   - Semantic insights across codebase
   - AI-generated summary
   - Actionable recommendations

## AI Integration

### LLM Usage Patterns:

1. **Strategy Generation**: 
   - Single call at initialization
   - Structured output (JSON)
   - ~500 tokens input, ~200 tokens output

2. **File Analysis**:
   - One call per file
   - Limited context (1-5KB)
   - ~1000 tokens input, ~300 tokens output

3. **Summary Generation**:
   - Single call at completion
   - Rich context from all analyses
   - ~2000 tokens input, ~200 tokens output

### Prompt Engineering Techniques:

1. **Role Definition**: "You are an expert file analysis strategist"
2. **Output Structuring**: Explicit JSON schema requirements
3. **Context Limiting**: Truncated content with clear markers
4. **Fallback Strategies**: Multiple parsing attempts with degradation

## Performance Optimizations

### 1. File Processing:
- **Parallel capability**: Can be extended for concurrent file analysis
- **Early termination**: Stops processing once sufficient high-relevance files found
- **Selective reading**: Only reads files passing initial filters

### 2. Vector Operations:
- **Batch embedding**: Processes multiple chunks simultaneously
- **Index optimization**: HNSW tuned for specific use case
- **Chunk caching**: Reuses embeddings for frequently accessed files

### 3. Memory Management:
- **Streaming reads**: Processes large files in chunks
- **Content truncation**: Limits AI context to manageable sizes
- **Result limiting**: Caps number of returned matches

### 4. Error Resilience:
- **Graceful degradation**: Falls back to heuristic analysis
- **Error collection**: Tracks but doesn't fail on individual file errors
- **Encoding flexibility**: Multiple encoding attempts per file

## Use Cases and Examples

### Example 1: Security Audit
```python
scraper = AgenticFileScraper(
    goal="Find potential SQL injection vulnerabilities and insecure password handling",
    ollama_model='codellama'
)

results = scraper.scan_directory(Path("/src/webapp"))

# Outputs:
# - Identifies files with raw SQL queries
# - Finds password hashing implementations
# - Detects missing input validation
# - Suggests parameterized queries and bcrypt usage
```

### Example 2: Documentation Analysis
```python
scraper = AgenticFileScraper(
    goal="Identify outdated or missing documentation for public APIs",
    ollama_model='llama3'
)

results = scraper.scan_directory(Path("/src/api"))

# Outputs:
# - Maps undocumented endpoints
# - Finds inconsistent documentation
# - Identifies deprecated features still documented
# - Recommends documentation updates
```

### Example 3: Performance Optimization
```python
scraper = AgenticFileScraper(
    goal="Find performance bottlenecks and inefficient algorithms",
    ollama_model='codellama'
)

results = scraper.scan_directory(Path("/src/core"))

# Outputs:
# - Detects N+1 query patterns
# - Identifies inefficient loops
# - Finds missing indexes
# - Suggests caching opportunities
```

## Advanced Features

### 1. Session Tracking:
- Each scan creates a unique session ID
- Enables comparison across multiple scans
- Tracks improvement over time

### 2. Metadata Enrichment:
- Stores file metadata alongside embeddings
- Enables filtered searches by file type, date, etc.
- Supports complex queries combining semantic and metadata filters

### 3. Incremental Updates:
- Can check if files already embedded
- Updates only changed files
- Maintains embedding version history

### 4. Cross-File Pattern Detection:
- Identifies similar code patterns across files
- Detects potential code duplication
- Finds inconsistent implementations of same logic

## Limitations and Considerations

### 1. Scalability:
- Embedding generation is CPU/GPU intensive
- Large codebases may require batch processing
- Vector database size grows with content

### 2. Accuracy:
- Dependent on LLM quality and prompt engineering
- May miss context-specific nuances
- Requires validation for critical findings

### 3. Privacy:
- Sends code snippets to LLM
- Stores embeddings persistently
- Should not be used with sensitive data without proper security

### 4. Maintenance:
- Embeddings should be regenerated when model updates
- Strategy patterns need periodic refinement
- LLM prompts may need adjustment for different domains

## Conclusion

The AgenticFileScraper and VectorDatabase combination provides a powerful system for intelligent code analysis that goes beyond simple pattern matching. By combining:

1. **AI-driven goal understanding**
2. **Intelligent file prioritization**
3. **Semantic similarity search**
4. **Context-aware analysis**
5. **Cross-file pattern detection**

The system enables developers to quickly understand large codebases, identify issues, and get actionable recommendations based on natural language objectives. The architecture is designed to be extensible, allowing for custom analysis strategies, different embedding models, and various LLM backends to suit specific use cases.