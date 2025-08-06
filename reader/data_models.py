#!/usr/bin/env python3
"""
Data models and structures for the agentic file scraper
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class FileAnalysis:
    """Results from analyzing a single file"""
    file_path: str
    file_type: str
    size_bytes: int
    relevance_score: float
    key_findings: List[str]
    content_summary: str
    recommendations: List[str]
    semantic_matches: List[Dict[str, Any]]
    analysis_timestamp: str = ""

    def __post_init__(self):
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def is_high_relevance(self, threshold: float = 0.6) -> bool:
        """Check if file has high relevance to the goal"""
        return self.relevance_score >= threshold

    def get_top_findings(self, limit: int = 3) -> List[str]:
        """Get top N key findings"""
        return self.key_findings[:limit]


@dataclass
class SemanticInsight:
    """Semantic insight from vector analysis"""
    query: str
    total_matches: int
    high_confidence_matches: int
    top_match_file: str
    top_match_content: str
    top_match_similarity: float
    related_files: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ScanStrategy:
    """Strategy for scanning based on parsed goal"""
    keywords: List[str]
    file_priorities: List[str]
    analysis_depth: str  # "shallow", "medium", "deep"
    patterns_to_find: List[str]
    scan_focus: str  # "security", "bugs", "performance", etc.
    semantic_queries: List[str]
    file_size_limit: int = 1024 * 1024  # 1MB default
    max_files_to_analyze: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def is_high_priority_file(self, file_path: str, file_extension: str) -> bool:
        """Check if file should be prioritized based on strategy"""
        # Check extension priority
        if file_extension.lower() in self.file_priorities:
            return True

        # Check filename keywords
        filename_lower = file_path.lower()
        for keyword in self.keywords:
            if keyword.lower() in filename_lower:
                return True

        return False


@dataclass
class ScanResults:
    """Complete scan results"""
    goal: str
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
    processing_time_seconds: float = 0.0

    def __post_init__(self):
        if not self.scan_timestamp:
            self.scan_timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert nested dataclasses
        result['strategy'] = self.strategy.to_dict()
        result['high_relevance_files'] = [f.to_dict() for f in self.high_relevance_files]
        result['semantic_insights'] = [s.to_dict() for s in self.semantic_insights]
        return result

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_files': self.total_files,
            'analyzed_files': self.analyzed_files,
            'skipped_files': self.skipped_files,
            'high_relevance_count': len(self.high_relevance_files),
            'semantic_insights_count': len(self.semantic_insights),
            'avg_relevance_score': self._calculate_avg_relevance(),
            'processing_time_seconds': self.processing_time_seconds,
            'files_per_second': self._calculate_files_per_second()
        }

    def _calculate_avg_relevance(self) -> float:
        """Calculate average relevance score"""
        if not self.high_relevance_files:
            return 0.0

        total_score = sum(f.relevance_score for f in self.high_relevance_files)
        return total_score / len(self.high_relevance_files)

    def _calculate_files_per_second(self) -> float:
        """Calculate processing speed"""
        if self.processing_time_seconds <= 0:
            return 0.0

        return self.analyzed_files / self.processing_time_seconds

    def get_top_files(self, limit: int = 10) -> List[FileAnalysis]:
        """Get top N most relevant files"""
        sorted_files = sorted(
            self.high_relevance_files,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        return sorted_files[:limit]

    def get_files_by_type(self) -> Dict[str, List[FileAnalysis]]:
        """Group files by file type"""
        files_by_type = {}
        for file_analysis in self.high_relevance_files:
            file_type = file_analysis.file_type
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file_analysis)

        return files_by_type

    def get_all_recommendations(self) -> List[str]:
        """Get all unique recommendations"""
        all_recs = set(self.recommendations)

        # Add recommendations from individual files
        for file_analysis in self.high_relevance_files:
            all_recs.update(file_analysis.recommendations)

        return list(all_recs)


@dataclass
class ProcessingError:
    """Error encountered during processing"""
    file_path: str
    error_type: str
    error_message: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ScanSession:
    """Information about a scan session stored in database"""
    session_id: str
    goal: str
    file_count: int
    findings_summary: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# Utility functions for data models

def create_file_analysis(file_path: str, file_type: str = "unknown",
                         size_bytes: int = 0, relevance_score: float = 0.0) -> FileAnalysis:
    """Create a FileAnalysis with default values"""
    return FileAnalysis(
        file_path=file_path,
        file_type=file_type,
        size_bytes=size_bytes,
        relevance_score=relevance_score,
        key_findings=[],
        content_summary="",
        recommendations=[],
        semantic_matches=[]
    )


def create_semantic_insight(query: str, matches: List[tuple]) -> Optional[SemanticInsight]:
    """Create SemanticInsight from search results"""
    if not matches:
        return None

    high_confidence = [m for m in matches if len(m) > 2 and m[2] > 0.7]

    top_match = matches[0]
    if len(top_match) < 3:
        return None

    related_files = []
    seen_files = set()

    for match in matches[:5]:  # Top 5 matches
        if len(match) >= 3:
            file_path, content, similarity = match[0], match[1], match[2]
            if file_path not in seen_files:
                related_files.append({
                    'file_path': file_path,
                    'similarity': similarity,
                    'content_preview': content[:100] + "..." if len(content) > 100 else content
                })
                seen_files.add(file_path)

    return SemanticInsight(
        query=query,
        total_matches=len(matches),
        high_confidence_matches=len(high_confidence),
        top_match_file=top_match[0],
        top_match_content=top_match[1][:200] + "...",
        top_match_similarity=top_match[2],
        related_files=related_files
    )


def create_default_strategy(goal: str) -> ScanStrategy:
    """Create a default strategy for a goal"""
    keywords = goal.lower().split()

    return ScanStrategy(
        keywords=keywords,
        file_priorities=[".py", ".js", ".java", ".cpp", ".c", ".md", ".txt", ".json", ".yaml", ".yml"],
        analysis_depth="medium",
        patterns_to_find=["issues", "problems", "improvements", "bugs", "errors"],
        scan_focus="general",
        semantic_queries=[goal]
    )