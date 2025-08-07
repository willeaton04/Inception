# 🔄 Vector Routing Adjustment Guide

The vector routing in your agentic file scraper controls how files are embedded, searched, and connected semantically. Here are the key areas you can adjust:

## 🎯 **1. Embedding Strategy (Text Chunking)**

### Current Implementation
```python
# In vector_db.py
def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100):
```

### Adjustments You Can Make

#### **A. Chunk Size Optimization**
```python
def chunk_text(self, text: str, max_chunk_size: int = 500, overlap: int = 50):
    """
    Smaller chunks = More granular semantic search
    Larger chunks = More context but less precise matches
    """
    
    # For code files - smaller chunks work better
    if file_path.suffix in ['.py', '.js', '.java', '.cpp']:
        max_chunk_size = 300  # Function-level granularity
        overlap = 30
    
    # For documentation - larger chunks preserve context
    elif file_path.suffix in ['.md', '.txt', '.rst']:
        max_chunk_size = 800  # Paragraph-level granularity
        overlap = 80
    
    # For config files - very small chunks
    elif file_path.suffix in ['.json', '.yaml', '.xml']:
        max_chunk_size = 200  # Property-level granularity
        overlap = 20
```

#### **B. Smart Boundary Detection**
```python
def chunk_text_smart(self, text: str, file_type: str) -> List[str]:
    """Chunk text based on content structure"""
    
    if file_type in ['text/python', 'text/javascript']:
        # Split by functions/classes
        return self._chunk_by_code_blocks(text)
    elif file_type == 'text/markdown':
        # Split by headers
        return self._chunk_by_headers(text)
    elif file_type == 'application/json':
        # Split by JSON objects
        return self._chunk_by_json_objects(text)
    else:
        # Default paragraph-based chunking
        return self._chunk_by_paragraphs(text)

def _chunk_by_code_blocks(self, code: str) -> List[str]:
    """Split code by functions, classes, and logical blocks"""
    chunks = []
    current_chunk = ""
    indent_level = 0
    
    for line in code.split('\n'):
        # Detect function/class definitions
        if re.match(r'^(def|class|function|var|const|let)\s+', line.strip()):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
            indent_level = len(line) - len(line.lstrip())
        else:
            current_chunk += line + '\n'
            
            # Split if chunk gets too large
            if len(current_chunk) > 800:
                chunks.append(current_chunk.strip())
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

## 🔍 **2. Search Strategy Adjustments**

### Current Implementation
```python
# In vector_db.py
def semantic_search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.3):
```

### Enhanced Search Routing
```python
def semantic_search_enhanced(self, query: str, top_k: int = 10, 
                           similarity_threshold: float = 0.4,
                           search_strategy: str = "balanced") -> List[Tuple[str, str, float]]:
    """Enhanced semantic search with multiple strategies"""
    
    if search_strategy == "precise":
        # High threshold, fewer results, high confidence
        similarity_threshold = 0.7
        top_k = 5
        
    elif search_strategy == "broad":
        # Low threshold, more results, discover connections
        similarity_threshold = 0.3
        top_k = 20
        
    elif search_strategy == "balanced":
        # Default balanced approach
        similarity_threshold = 0.5
        top_k = 10
    
    # Execute search with strategy
    raw_results = self._execute_search(query, top_k, similarity_threshold)
    
    # Post-process based on strategy
    return self._post_process_results(raw_results, search_strategy)

def _post_process_results(self, results: List, strategy: str) -> List:
    """Post-process search results based on strategy"""
    
    if strategy == "precise":
        # Group by file and return best match per file
        file_groups = {}
        for file_path, content, similarity in results:
            if file_path not in file_groups or similarity > file_groups[file_path][2]:
                file_groups[file_path] = (file_path, content, similarity)
        return list(file_groups.values())
    
    elif strategy == "broad":
        # Return all results, but diversify by file type
        file_types = {}
        diverse_results = []
        
        for result in results:
            file_ext = Path(result[0]).suffix
            if file_types.get(file_ext, 0) < 3:  # Max 3 per file type
                diverse_results.append(result)
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        return diverse_results
    
    return results
```

## 🎯 **3. Query Generation & Routing**

### Current Implementation
```python
# In agentic_scraper.py
semantic_queries = strategy.semantic_queries  # ["find security issues"]
```

### Enhanced Query Generation
```python
def generate_enhanced_queries(self, goal: str, strategy: ScanStrategy) -> List[Dict[str, Any]]:
    """Generate multiple types of semantic queries"""
    
    queries = []
    
    # 1. Direct goal query
    queries.append({
        'query': goal,
        'type': 'primary',
        'weight': 1.0,
        'strategy': 'balanced'
    })
    
    # 2. Keyword expansion queries
    for keyword in strategy.keywords:
        queries.append({
            'query': f"{keyword} implementation patterns",
            'type': 'keyword',
            'weight': 0.8,
            'strategy': 'broad'
        })
    
    # 3. Pattern-specific queries
    for pattern in strategy.patterns_to_find:
        queries.append({
            'query': f"examples of {pattern}",
            'type': 'pattern',
            'weight': 0.9,
            'strategy': 'precise'
        })
    
    # 4. Context queries (what's related)
    queries.append({
        'query': f"files related to {goal}",
        'type': 'context',
        'weight': 0.6,
        'strategy': 'broad'
    })
    
    return queries

def run_enhanced_semantic_analysis(self, files: List[Path]) -> List[SemanticInsight]:
    """Run semantic analysis with enhanced query routing"""
    
    enhanced_queries = self.generate_enhanced_queries(self.goal, self.strategy)
    insights = []
    
    for query_info in enhanced_queries:
        print(f'\033[1;90m  Query ({query_info["type"]}): "{query_info["query"]}"\033[0m')
        
        # Use different search strategies per query type
        matches = self.vector_db.semantic_search_enhanced(
            query_info['query'],
            top_k=15,
            search_strategy=query_info['strategy']
        )
        
        if matches:
            insight = create_semantic_insight(query_info['query'], matches)
            if insight:
                # Weight the insight based on query importance
                insight.confidence_weight = query_info['weight']
                insights.append(insight)
    
    return self._consolidate_insights(insights)
```

## 🚀 **4. File-Type Specific Routing**

### Current Implementation
```python
# Generic embedding for all files
self.vector_db.embed_file(file_path, content, metadata)
```

### File-Type Aware Routing
```python
def embed_file_smart(self, file_path: Path, content: str, metadata: Dict = None) -> bool:
    """Smart embedding based on file type"""
    
    file_ext = file_path.suffix.lower()
    
    # Route to specialized embedding strategies
    if file_ext in ['.py', '.js', '.java', '.cpp', '.c']:
        return self._embed_code_file(file_path, content, metadata)
    elif file_ext in ['.md', '.txt', '.rst']:
        return self._embed_document_file(file_path, content, metadata)
    elif file_ext in ['.json', '.yaml', '.xml']:
        return self._embed_config_file(file_path, content, metadata)
    else:
        return self._embed_generic_file(file_path, content, metadata)

def _embed_code_file(self, file_path: Path, content: str, metadata: Dict) -> bool:
    """Specialized embedding for code files"""
    
    # Extract code structure
    functions = self._extract_functions(content, file_path.suffix)
    classes = self._extract_classes(content, file_path.suffix)
    imports = self._extract_imports(content, file_path.suffix)
    
    # Create specialized chunks
    chunks = []
    
    # Function-level chunks
    for func in functions:
        chunks.append({
            'content': func['code'],
            'type': 'function',
            'name': func['name'],
            'metadata': {**metadata, 'function_name': func['name']}
        })
    
    # Class-level chunks  
    for cls in classes:
        chunks.append({
            'content': cls['code'],
            'type': 'class',
            'name': cls['name'],
            'metadata': {**metadata, 'class_name': cls['name']}
        })
    
    # Import/dependency chunk
    if imports:
        chunks.append({
            'content': '\n'.join(imports),
            'type': 'imports',
            'metadata': {**metadata, 'dependencies': len(imports)}
        })
    
    return self._embed_structured_chunks(file_path, chunks)

def _embed_document_file(self, file_path: Path, content: str, metadata: Dict) -> bool:
    """Specialized embedding for documentation"""
    
    # Extract document structure
    headers = self._extract_headers(content)
    
    chunks = []
    current_section = ""
    current_header = "Introduction"
    
    for line in content.split('\n'):
        if line.startswith('#'):
            # New section found
            if current_section:
                chunks.append({
                    'content': current_section,
                    'type': 'section',
                    'header': current_header,
                    'metadata': {**metadata, 'section': current_header}
                })
            current_header = line.strip('#').strip()
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    # Add final section
    if current_section:
        chunks.append({
            'content': current_section,
            'type': 'section',
            'header': current_header,
            'metadata': {**metadata, 'section': current_header}
        })
    
    return self._embed_structured_chunks(file_path, chunks)
```

## 🎛️ **5. Similarity Threshold Tuning**

### Dynamic Threshold Adjustment
```python
def calculate_dynamic_threshold(self, query: str, file_types: List[str]) -> float:
    """Calculate optimal similarity threshold based on context"""
    
    base_threshold = 0.5
    
    # Adjust based on query specificity
    query_words = len(query.split())
    if query_words <= 2:
        # Very specific queries need higher threshold
        base_threshold += 0.2
    elif query_words >= 6:
        # Broad queries can use lower threshold
        base_threshold -= 0.1
    
    # Adjust based on file types being searched
    if any(ext in ['.py', '.js', '.java'] for ext in file_types):
        # Code files often have exact matches, use higher threshold
        base_threshold += 0.1
    elif any(ext in ['.md', '.txt'] for ext in file_types):
        # Documentation can be more varied, use lower threshold
        base_threshold -= 0.1
    
    return max(0.3, min(0.8, base_threshold))  # Clamp between 0.3 and 0.8
```

## 🔧 **6. Integration Points in agentic_scraper.py**

### Update the analyze_file method
```python
def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
    # ... existing code ...
    
    # Use enhanced embedding
    embedding_success = self.vector_db.embed_file_smart(file_path, content, metadata)
    
    # Use enhanced semantic search
    semantic_matches = []
    for query_info in self.generate_enhanced_queries(self.goal, self.strategy):
        matches = self.vector_db.semantic_search_enhanced(
            query_info['query'],
            top_k=3,
            file_filter=[str(file_path)],
            search_strategy=query_info['strategy']
        )
        # Process matches...
    
    # ... rest of method
```

## 📊 **7. Performance Optimizations**

### Batch Processing
```python
def embed_files_batch(self, files_and_content: List[Tuple[Path, str]]) -> Dict[Path, bool]:
    """Embed multiple files efficiently"""
    
    # Group by file type for batch processing
    by_type = {}
    for file_path, content in files_and_content:
        ext = file_path.suffix.lower()
        if ext not in by_type:
            by_type[ext] = []
        by_type[ext].append((file_path, content))
    
    results = {}
    
    # Process each file type with optimized strategy
    for file_type, files in by_type.items():
        if file_type in ['.py', '.js', '.java']:
            # Use code-optimized embedding
            batch_results = self._batch_embed_code_files(files)
        elif file_type in ['.md', '.txt']:
            # Use document-optimized embedding
            batch_results = self._batch_embed_document_files(files)
        else:
            # Use generic embedding
            batch_results = self._batch_embed_generic_files(files)
        
        results.update(batch_results)
    
    return results
```

### Caching Strategy
```python
def get_cached_similarity(self, query_hash: str, file_path: str) -> Optional[float]:
    """Cache similarity calculations for repeated queries"""
    
    cache_key = f"{query_hash}:{file_path}"
    
    # Check in-memory cache first
    if hasattr(self, '_similarity_cache') and cache_key in self._similarity_cache:
        return self._similarity_cache[cache_key]
    
    # Check database cache
    cached_result = self._get_from_db_cache(cache_key)
    if cached_result:
        return cached_result
    
    return None

def cache_similarity(self, query_hash: str, file_path: str, similarity: float):
    """Cache similarity result for future use"""
    
    # Store in memory cache
    if not hasattr(self, '_similarity_cache'):
        self._similarity_cache = {}
    
    cache_key = f"{query_hash}:{file_path}"
    self._similarity_cache[cache_key] = similarity
    
    # Store in database cache for persistence
    self._store_in_db_cache(cache_key, similarity)
```

## 🎯 **Quick Adjustment Recommendations**

### For Better Semantic Search:
1. **Reduce chunk size** to 300-500 chars for more precise matches
2. **Lower similarity threshold** to 0.4 for broader discovery
3. **Add file-type specific queries** based on scan focus

### For Faster Processing:
1. **Increase similarity threshold** to 0.6+ for fewer matches
2. **Limit top_k** to 5-8 results per query
3. **Use batch embedding** for multiple files

### For Better Code Analysis:
1. **Function-level chunking** for code files
2. **Import/dependency specific queries**
3. **Higher precision thresholds** (0.7+) for exact matches

### For Better Documentation Discovery:
1. **Section-based chunking** for markdown files
2. **Lower thresholds** (0.3-0.4) for concept discovery  
3. **Header-aware semantic queries**

The key is to **experiment with these settings** based on your specific use cases and the types of insights you want to discover! 🎯