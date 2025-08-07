#!/usr/bin/env python3
"""
Data models for the Agentic Question Answering System
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class ScanStrategy:
    """Strategy for scanning and analyzing files"""
    keywords: List[str] = field(default_factory=list)
    file_priorities: List[str] = field(default_factory=list)
    analysis_depth: str = "medium"  # shallow, medium, deep
    patterns_to_find: List[str] = field(default_factory=list)
    scan_focus: str = "general"  # security, bugs, performance, documentation, structure, general
    semantic_queries: List[str] = field(default_factory=list)
    max_files_to_analyze: int = 50
    file_size_limit: int = 10 * 1024 * 1024  # 10MB default

    def is_high_priority_file(self, file_path: str, extension: str) -> bool:
        """Check if file is high priority based on name patterns"""
        high_priority_names = [
            'main', 'index', 'app', 'config', 'settings',
            'auth', 'security', 'api', 'database', 'model'
        ]
        file_lower = file_path.lower()
        return any(name in file_lower for name in high_priority_names)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    file_type: str
    size_bytes: int
    relevance_score: float
    key_findings: List[str]
    content_summary: str
    recommendations: List[str]
    semantic_matches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SemanticInsight:
    """Semantic insight from vector search"""
    query: str
    matches: List[tuple]  # (file_path, content, similarity)
    high_confidence_matches: int = 0
    average_similarity: float = 0.0
    key_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate metrics after initialization"""
        if self.matches:
            similarities = [match[2] for match in self.matches]
            self.average_similarity = sum(similarities) / len(similarities)
            self.high_confidence_matches = sum(1 for s in similarities if s > 0.7)


@dataclass
class ProcessingError:
    """Error that occurred during processing"""
    file_path: str
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_path': self.file_path,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ScanResults:
    """Complete results from a scanning session"""
    goal: str  # Keep as 'goal' for backward compatibility
    session_id: str
    strategy: ScanStrategy
    total_files: int
    analyzed_files: int
    skipped_files: int
    high_relevance_files: List[FileAnalysis]
    semantic_insights: List[SemanticInsight]
    summary: str
    recommendations: List[str]
    scan_timestamp: str
    processing_time_seconds: float
    processing_errors: List[ProcessingError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'goal': self.goal,
            'session_id': self.session_id,
            'strategy': self.strategy.to_dict(),
            'total_files': self.total_files,
            'analyzed_files': self.analyzed_files,
            'skipped_files': self.skipped_files,
            'high_relevance_files': [f.to_dict() for f in self.high_relevance_files],
            'semantic_insights': len(self.semantic_insights),
            'summary': self.summary,
            'recommendations': self.recommendations,
            'scan_timestamp': self.scan_timestamp,
            'processing_time_seconds': self.processing_time_seconds,
            'processing_errors': [e.to_dict() for e in self.processing_errors]
        }


def create_default_strategy(goal_or_question: str) -> ScanStrategy:
    """Create a default strategy based on the goal/question"""
    goal_lower = goal_or_question.lower()

    # Default keywords from the goal
    keywords = [word for word in goal_or_question.split()
                if len(word) > 3 and word.isalnum()]

    # Determine focus based on keywords
    if any(word in goal_lower for word in ['security', 'vulnerability', 'auth', 'password']):
        scan_focus = 'security'
        file_priorities = ['.py', '.js', '.java', '.php', '.rb']
    elif any(word in goal_lower for word in ['bug', 'error', 'issue', 'problem', 'fix']):
        scan_focus = 'bugs'
        file_priorities = ['.py', '.js', '.java', '.log', '.txt']
    elif any(word in goal_lower for word in ['performance', 'speed', 'optimize', 'slow']):
        scan_focus = 'performance'
        file_priorities = ['.py', '.js', '.java', '.sql', '.yaml']
    elif any(word in goal_lower for word in ['document', 'readme', 'guide', 'explain']):
        scan_focus = 'documentation'
        file_priorities = ['.md', '.txt', '.rst', '.html', '.pdf']
    else:
        scan_focus = 'general'
        file_priorities = ['.py', '.js', '.java', '.md', '.txt']

    return ScanStrategy(
        keywords=keywords[:10],  # Limit keywords
        file_priorities=file_priorities,
        analysis_depth='medium',
        patterns_to_find=[],
        scan_focus=scan_focus,
        semantic_queries=[goal_or_question],
        max_files_to_analyze=50
    )


def create_file_analysis(file_path: str, file_type: str, size: int,
                         relevance: float = 0.5) -> FileAnalysis:
    """Create a basic file analysis object"""
    return FileAnalysis(
        file_path=file_path,
        file_type=file_type,
        size_bytes=size,
        relevance_score=relevance,
        key_findings=[],
        content_summary="",
        recommendations=[]
    )


def create_semantic_insight(query: str, matches: List[tuple]) -> Optional[SemanticInsight]:
    """Create a semantic insight from search results"""
    if not matches:
        return None

    return SemanticInsight(
        query=query,
        matches=matches,
        key_patterns=[]
    )