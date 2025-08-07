# 📊 Data Models Architecture (`data_models.py`)

The `data_models.py` file is the **data backbone** of the agentic file scraper. It defines all the structured data types that flow through the system, from individual file analysis to complete scan results.

## 🏗️ **Core Architecture Pattern**

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
```

Uses Python **dataclasses** for:
- Automatic `__init__`, `__repr__`, `__eq__` methods
- Type hints for better IDE support and validation
- Clean, readable data structure definitions
- Easy serialization to JSON/dict formats

## 📋 **Data Flow Architecture**

```
Goal Input → ScanStrategy → FileAnalysis → SemanticInsight → ScanResults
     ↓             ↓            ↓              ↓             ↓
  Natural      AI-parsed    Per-file      Cross-file    Final Report
  Language     Strategy     Analysis      Patterns      & Summary
```

## 🎯 **Core Data Models**

### **1. FileAnalysis** - Individual File Results
```python
@dataclass
class FileAnalysis:
    file_path: str                    # "/path/to/file.py"
    file_type: str                    # "text/python"
    size_bytes: int                   # 15420
    relevance_score: float            # 0.85 (0-1 scale)
    key_findings: List[str]           # ["Security issue found", "TODO items"]
    content_summary: str              # "Python module with auth functions"
    recommendations: List[str]        # ["Review auth logic", "Add tests"]
    semantic_matches: List[Dict]      # Vector search matches
    analysis_timestamp: str           # ISO timestamp
```

**Purpose**: Stores everything discovered about a single file
**Used by**: `AgenticFileScraper.analyze_file()` method
**Key Features**:
- `relevance_score` determines if file is "high relevance" (>0.6)
- `semantic_matches` connects to vector database results
- Automatic timestamp generation via `__post_init__`

### **2. ScanStrategy** - AI-Generated Scanning Plan
```python
@dataclass
class ScanStrategy:
    keywords: List[str]               # ["security", "auth", "password"]
    file_priorities: List[str]        # [".py", ".js", ".php"]
    analysis_depth: str               # "shallow", "medium", "deep"
    patterns_to_find: List[str]       # ["hardcoded credentials", "SQL injection"]
    scan_focus: str                   # "security", "bugs", "performance"
    semantic_queries: List[str]       # ["authentication vulnerabilities"]
    file_size_limit: int = 1MB        # Maximum file size to analyze
    max_files_to_analyze: int = 50    # Performance limit
```

**Purpose**: Converts natural language goals into actionable scanning parameters
**Created by**: `AgenticFileScraper._parse_goal()` using AI
**Key Features**:
- `is_high_priority_file()` method for file prioritization
- Configurable limits for performance control
- Drives both file selection and AI analysis prompts

### **3. SemanticInsight** - Cross-File Pattern Discovery
```python
@dataclass
class SemanticInsight:
    query: str                        # "authentication vulnerabilities"
    total_matches: int                # 15
    high_confidence_matches: int      # 7 (similarity > 0.7)
    top_match_file: str              # "auth.py"
    top_match_content: str           # "def authenticate(username, password)..."
    top_match_similarity: float      # 0.92
    related_files: List[Dict]        # Other relevant files
```

**Purpose**: Captures patterns found across multiple files using vector search
**Created by**: `create_semantic_insight()` from vector database results
**Key Features**:
- Identifies code patterns that span multiple files
- Confidence scoring for reliability
- Links related files by semantic similarity

### **4. ScanResults** - Complete Analysis Output
```python
@dataclass
class ScanResults:
    goal: str                        # Original user goal
    session_id: str                  # Unique scan session
    strategy: ScanStrategy           # How the scan was executed
    total_files: int                 # Files found
    analyzed_files: int              # Files successfully processed
    skipped_files: int               # Files ignored (too large, etc.)
    high_relevance_files: List[FileAnalysis]  # Score > 0.6
    semantic_insights: List[SemanticInsight]  # Cross-file patterns
    summary: str                     # AI-generated overview
    recommendations: List[str]        # Actionable next steps
    scan_timestamp: str              # When scan completed
    processing_time_seconds: float   # Performance metrics
```

**Purpose**: Complete scan results for reporting and analysis
**Created by**: `AgenticFileScraper.scan_directory()`
**Key Features**:
- `to_dict()` method for JSON serialization
- `get_summary_stats()` for analytics
- Rich metadata for audit trails

## 🔧 **Utility Functions**

### **Helper Functions for Data Creation**
```python
def create_file_analysis(file_path: str, file_type: str = "unknown", 
                        size_bytes: int = 0, relevance_score: float = 0.0) -> FileAnalysis:
    """Create FileAnalysis with defaults"""

def create_semantic_insight(query: str, matches: List[tuple]) -> Optional[SemanticInsight]:
    """Convert vector search results into SemanticInsight"""

def create_default_strategy(goal: str) -> ScanStrategy:
    """Fallback strategy when AI parsing fails"""
```

**Purpose**: Factory functions for safe object creation
**Usage**: Provide defaults and validation when creating data objects

## 📊 **Advanced Features**

### **1. Automatic Post-Processing**
```python
def __post_init__(self):
    if not self.analysis_timestamp:
        self.analysis_timestamp = datetime.now().isoformat()
```
- Automatically adds timestamps
- Validates required fields
- Normalizes data formats

### **2. Business Logic Methods**
```python
def is_high_relevance(self, threshold: float = 0.6) -> bool:
    """Check if file meets relevance threshold"""

def get_top_files(self, limit: int = 10) -> List[FileAnalysis]:
    """Get most relevant files, sorted by score"""

def get_files_by_type(self) -> Dict[str, List[FileAnalysis]]:
    """Group results by file extension"""
```

**Purpose**: Encapsulate common operations on data
**Benefits**: Keeps logic close to data, makes code more maintainable

### **3. Serialization Support**
```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for JSON export"""
    result = asdict(self)
    # Handle nested dataclasses
    result['strategy'] = self.strategy.to_dict()
    result['high_relevance_files'] = [f.to_dict() for f in self.high_relevance_files]
    return result
```

**Purpose**: Clean JSON export for APIs, file storage, web interfaces
**Challenge**: Handles nested dataclass serialization properly

## 🔄 **Data Flow Example**

### **1. Input Processing**
```python
# User input: "find security vulnerabilities"
goal = "find security vulnerabilities"
```

### **2. Strategy Generation**
```python
strategy = ScanStrategy(
    keywords=["security", "vulnerabilities", "auth"],
    file_priorities=[".py", ".js", ".php"],
    scan_focus="security",
    semantic_queries=["security flaws", "authentication issues"]
)
```

### **3. File Analysis Loop**
```python
for file_path in files:
    analysis = FileAnalysis(
        file_path=str(file_path),
        relevance_score=0.85,
        key_findings=["Hardcoded password found", "SQL injection risk"],
        recommendations=["Use environment variables", "Add input validation"]
    )
    analyses.append(analysis)
```

### **4. Semantic Analysis**
```python
insight = SemanticInsight(
    query="authentication vulnerabilities",
    total_matches=12,
    high_confidence_matches=5,
    top_match_file="auth.py",
    top_match_similarity=0.91
)
```

### **5. Final Results**
```python
results = ScanResults(
    goal=goal,
    strategy=strategy,
    high_relevance_files=high_scoring_files,
    semantic_insights=[insight],
    summary="Found 5 security issues across 12 files...",
    recommendations=["Audit authentication logic", "Add input validation"]
)
```

## 🎯 **Key Design Benefits**

### **1. Type Safety**
- Type hints catch errors at development time
- IDE provides better autocomplete and validation
- Runtime type checking possible with additional tools

### **2. Immutability by Default**
- Dataclasses encourage immutable data structures
- Reduces bugs from unexpected state changes
- Makes concurrent processing safer

### **3. Serialization Ready**
- Easy conversion to JSON for APIs
- Clean dictionary format for templates
- Database storage compatibility

### **4. Extensible**
- Easy to add new fields without breaking existing code
- Optional fields with defaults
- Custom methods can be added to any dataclass

### **5. Self-Documenting**
- Field types and names explain data structure
- Docstrings provide usage context
- Example values in type hints guide usage

## 🚀 **Usage Patterns**

### **Creating Objects**
```python
# Using factory functions (recommended)
analysis = create_file_analysis("main.py", "text/python", 1500, 0.8)

# Direct instantiation
analysis = FileAnalysis(
    file_path="main.py",
    file_type="text/python", 
    size_bytes=1500,
    relevance_score=0.8,
    # ... other required fields
)
```

### **Working with Results**
```python
# Filter high relevance
high_rel = [f for f in results.high_relevance_files if f.is_high_relevance()]

# Get summary stats
stats = results.get_summary_stats()
print(f"Found {stats['high_relevance_count']} important files")

# Export to JSON
json_data = results.to_dict()
```

### **Processing Data**
```python
# Group by file type
by_type = results.get_files_by_type()
python_files = by_type.get('.py', [])

# Sort by relevance  
sorted_files = sorted(analyses, key=lambda x: x.relevance_score, reverse=True)
```

The data models serve as the **contract** between different parts of the system, ensuring consistent data flow from AI analysis through vector processing to final reporting. They make the codebase more maintainable, testable, and extensible! 🎯