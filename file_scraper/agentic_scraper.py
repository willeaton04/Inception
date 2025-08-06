#!/usr/bin/env python3
"""
Main agentic file scraper implementation
"""

import json
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from agent.ollama_client import OllamaManager
from db.vector_db import VectorDatabase
from reader.data_models import (
    FileAnalysis, SemanticInsight, ScanStrategy, ScanResults, ProcessingError,
    create_file_analysis, create_semantic_insight, create_default_strategy
)


class AgenticFileScraper:
    """Main orchestrator for agentic file scraping with AI and vector context"""

    def __init__(self, goal: str, ollama_model: str = 'llama3.1', vector_db_path: str = "file_vectors.db"):
        self.goal = goal
        self.start_time = time.time()

        # Generate session ID
        self.session_id = hashlib.md5(f"{goal}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Initialize components
        self.ollama = OllamaManager(model=ollama_model)
        self.vector_db = VectorDatabase(db_path=vector_db_path)

        # Track processing errors
        self.processing_errors: List[ProcessingError] = []

        # Initialize Ollama
        if not self.ollama.start_ollama():
            raise RuntimeError("Failed to start Ollama - check installation and model availability")

        print(f'\033[1;32m[Scraper Initialized]:\033[0m Session {self.session_id}')

        # Parse goal into strategy
        self.strategy = self._parse_goal()

    def _parse_goal(self) -> ScanStrategy:
        """Parse goal into actionable strategy using AI"""
        system_prompt = "You are an expert analyst. Create a concise 2-3 sentence summary of file scan results."

        prompt = f"""
Scan Goal: {self.goal}
Focus Area: {self.strategy.scan_focus}
Files Analyzed: {len(analyses)}
High Relevance Files: {high_relevance_count}
Average Relevance: {avg_relevance:.2f}
Semantic Insights: {len(semantic_insights)}
Top Findings: {top_findings[:6]}

Write a concise summary highlighting the most important discoveries:"""

        ai_summary = self.ollama.generate(prompt, system_prompt)

        if ai_summary:
            return ai_summary
        else:
            # Fallback summary
            return f"Analyzed {len(analyses)} files with {high_relevance_count} highly relevant to '{self.goal}'. Found {len(semantic_insights)} semantic patterns across the codebase."

    def _generate_recommendations(self, analyses: List[FileAnalysis], semantic_insights: List[SemanticInsight]) -> List[
        str]:
        """Generate comprehensive recommendations"""
        recommendations = set()

        # Collect recommendations from file analyses
        for analysis in analyses:
            for rec in analysis.recommendations:
                if rec and len(rec.strip()) > 10:  # Filter out empty/short recs
                    recommendations.add(rec)

        # Add strategy-based recommendations
        if self.strategy.scan_focus == 'security':
            recommendations.add("Review authentication and authorization mechanisms")
            recommendations.add("Audit for hardcoded credentials and secrets")
        elif self.strategy.scan_focus == 'bugs':
            recommendations.add("Run automated testing on identified problem areas")
            recommendations.add("Consider code review for high-complexity functions")
        elif self.strategy.scan_focus == 'performance':
            recommendations.add("Profile identified performance bottlenecks")
            recommendations.add("Consider caching strategies for frequently accessed data")

        # Add semantic-based recommendations
        if semantic_insights:
            high_similarity_files = [s for s in semantic_insights if s.high_confidence_matches > 0]
            if high_similarity_files:
                recommendations.add("Review files with high semantic similarity for consistency")

        # Convert to list and limit
        rec_list = list(recommendations)
        return rec_list[:8] if rec_list else ["Continue monitoring files for changes"]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing session"""
        return {
            'session_id': self.session_id,
            'goal': self.goal,
            'strategy': self.strategy.to_dict(),
            'processing_errors': [error.to_dict() for error in self.processing_errors],
            'total_processing_time': time.time() - self.start_time,
            'vector_db_stats': self.vector_db.get_database_stats()
        }
        """You are an expert file analysis strategist. Parse the user's goal into a JSON strategy.

Return ONLY valid JSON with these exact fields:
- keywords: List of important search terms (3-10 items)
- file_priorities: List of file extensions to prioritize (e.g., [".py", ".js"])  
- analysis_depth: Either "shallow", "medium", or "deep"
- patterns_to_find: List of specific patterns to look for (3-8 items)
- scan_focus: One of "security", "bugs", "performance", "documentation", "structure", or "general"
- semantic_queries: List of 2-5 semantic search queries to run

Example output:
{
  "keywords": ["security", "auth", "login", "password"],
  "file_priorities": [".py", ".js", ".php", ".java"],
  "analysis_depth": "deep",
  "patterns_to_find": ["hardcoded credentials", "SQL injection", "XSS vulnerabilities"],
  "scan_focus": "security",
  "semantic_queries": ["authentication vulnerabilities", "security flaws", "access control issues"]
}"""

        prompt = f'Goal: "{self.goal}"\n\nAnalyze this goal and create a scanning strategy:'

        response = self.ollama.generate(prompt, system_prompt)

        try:
            # Extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                strategy_data = json.loads(response[start:end])

                strategy = ScanStrategy(
                    keywords=strategy_data.get('keywords', []),
                    file_priorities=strategy_data.get('file_priorities', []),
                    analysis_depth=strategy_data.get('analysis_depth', 'medium'),
                    patterns_to_find=strategy_data.get('patterns_to_find', []),
                    scan_focus=strategy_data.get('scan_focus', 'general'),
                    semantic_queries=strategy_data.get('semantic_queries', [self.goal])
                )

                print(f'\033[1;34m[Strategy]:\033[0m Focus: {strategy.scan_focus} | Depth: {strategy.analysis_depth}')
                return strategy

        except (json.JSONDecodeError, KeyError) as e:
            print(f'\033[1;33m[Strategy Warning]:\033[0m Using fallback strategy: {str(e)}')

        # Fallback strategy
        return create_default_strategy(self.goal)

    def find_files(self, path: Path, extensions: List[str] = None) -> List[Path]:
        """Find files with intelligent prioritization based on strategy"""
        files = []

        # Use strategy priorities if no extensions specified
        target_extensions = extensions or self.strategy.file_priorities

        if path.is_file():
            if self._should_include_file(path, target_extensions):
                files.append(path)
        else:
            try:
                for file_path in path.rglob('*'):
                    if file_path.is_file() and self._should_include_file(file_path, target_extensions):
                        files.append(file_path)

            except PermissionError:
                error = ProcessingError(
                    file_path=str(path),
                    error_type="PermissionError",
                    error_message="Permission denied accessing directory"
                )
                self.processing_errors.append(error)
                print(f'\033[1;33m[Warning]:\033[0m Permission denied accessing some files in {path}')

        # Prioritize files based on strategy
        return self._prioritize_files(files)

    def _should_include_file(self, file_path: Path, extensions: List[str]) -> bool:
        """Determine if file should be included based on filters"""
        # Skip hidden files and common ignore patterns
        if any(part.startswith('.') for part in file_path.parts):
            return False

        # Skip common build/cache directories
        ignore_patterns = ['node_modules', '__pycache__', '.git', 'build', 'dist', 'target', 'bin', 'obj']
        if any(ignore in str(file_path) for ignore in ignore_patterns):
            return False

        # Check file size limit
        try:
            if file_path.stat().st_size > self.strategy.file_size_limit:
                return False
        except OSError:
            return False

        # Check extension filter
        if extensions and file_path.suffix.lower() not in extensions:
            return False

        return True

    def _prioritize_files(self, files: List[Path]) -> List[Path]:
        """Prioritize files based on strategy and goal relevance"""

        def priority_score(file_path: Path) -> float:
            score = 0.0

            # Strategy-based scoring
            if self.strategy.is_high_priority_file(str(file_path), file_path.suffix):
                score += 15

            # Extension priority from strategy
            if file_path.suffix.lower() in self.strategy.file_priorities:
                score += 10

            # Keyword matching in filename/path
            path_str = str(file_path).lower()
            for keyword in self.strategy.keywords:
                if keyword.lower() in path_str:
                    score += 5

            # Pattern matching for focus area
            focus_keywords = {
                'security': ['auth', 'login', 'password', 'secret', 'key', 'token', 'crypto'],
                'bugs': ['test', 'debug', 'error', 'exception', 'bug', 'issue'],
                'performance': ['cache', 'optimize', 'performance', 'speed', 'memory'],
                'documentation': ['readme', 'doc', 'guide', 'manual', 'help'],
                'structure': ['config', 'setup', 'init', 'main', 'core', 'base']
            }

            if self.strategy.scan_focus in focus_keywords:
                for keyword in focus_keywords[self.strategy.scan_focus]:
                    if keyword in path_str:
                        score += 3

            # File size scoring (prefer moderate sizes)
            try:
                size = file_path.stat().st_size
                if 1024 < size < 1024 * 50:  # 1KB - 50KB sweet spot
                    score += 4
                elif size < 1024 * 200:  # Up to 200KB
                    score += 2
                elif size < 1024 * 500:  # Up to 500KB
                    score += 1
            except OSError:
                pass

            return score

        return sorted(files, key=priority_score, reverse=True)

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze single file with AI and vector context"""
        try:
            # Read file content safely
            content = self._read_file_safely(file_path)
            if not content:
                return None

            # Get file metadata
            file_stats = file_path.stat()
            file_type = mimetypes.guess_type(str(file_path))[0] or "unknown"

            # Embed file content in vector database
            metadata = {
                'file_type': file_type,
                'size': file_stats.st_size,
                'modified': file_stats.st_mtime,
                'scan_session': self.session_id
            }

            embedding_success = self.vector_db.embed_file(file_path, content, metadata)
            if not embedding_success:
                print(f'\033[1;33m[Warning]:\033[0m Failed to embed {file_path.name}')

            # Find semantic matches with goal-based queries
            semantic_matches = []
            for query in self.strategy.semantic_queries:
                matches = self.vector_db.semantic_search(
                    query,
                    top_k=3,
                    file_filter=[str(file_path)],
                    similarity_threshold=0.6
                )

                for match_path, match_content, similarity in matches:
                    semantic_matches.append({
                        'query': query,
                        'content': match_content[:200] + "...",
                        'similarity': similarity
                    })

            # AI analysis with context
            analysis_data = self._ai_analyze_file(file_path, content, file_type, semantic_matches)

            return FileAnalysis(
                file_path=str(file_path),
                file_type=file_type,
                size_bytes=file_stats.st_size,
                relevance_score=analysis_data['relevance_score'],
                key_findings=analysis_data['key_findings'],
                content_summary=analysis_data['content_summary'],
                recommendations=analysis_data['recommendations'],
                semantic_matches=semantic_matches
            )

        except Exception as e:
            error = ProcessingError(
                file_path=str(file_path),
                error_type=type(e).__name__,
                error_message=str(e)
            )
            self.processing_errors.append(error)
            print(f'\033[1;33m[Warning]:\033[0m Error analyzing {file_path.name}: {str(e)}')
            return None

    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read file with multiple encoding attempts"""
        try:
            # Check size limit
            if file_path.stat().st_size > self.strategy.file_size_limit:
                return ""
        except OSError:
            return ""

        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Basic content validation
                    if len(content.strip()) < 10:  # Skip nearly empty files
                        return ""
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                break

        return ""

    def _ai_analyze_file(self, file_path: Path, content: str, file_type: str,
                         semantic_matches: List[Dict]) -> Dict[str, Any]:
        """AI-powered file analysis with context"""
        # Prepare content for analysis (truncate if necessary)
        analysis_content = content
        max_content_length = {
            'shallow': 2000,
            'medium': 5000,
            'deep': 8000
        }.get(self.strategy.analysis_depth, 5000)

        if len(content) > max_content_length:
            analysis_content = content[:max_content_length] + "\n... (truncated for analysis)"

        # Build context from semantic matches
        context_info = ""
        if semantic_matches:
            context_info = "\n\nSemantic context (related content found):\n"
            for match in semantic_matches[:2]:  # Limit context
                context_info += f"- Query '{match['query']}': {match['content']} (similarity: {match['similarity']:.2f})\n"

        # Build analysis prompt based on strategy
        system_prompt = f"""You are an expert file analyzer focusing on {self.strategy.scan_focus}. 

Goal: {self.goal}
Analysis Depth: {self.strategy.analysis_depth}
Looking for: {', '.join(self.strategy.patterns_to_find)}
Key concepts: {', '.join(self.strategy.keywords)}

Analyze this file and return ONLY valid JSON with these exact fields:
- relevance_score: Float between 0-1 (how relevant to the goal)
- key_findings: Array of 2-5 important discoveries related to the goal
- content_summary: String summarizing what this file contains (1-2 sentences)
- recommendations: Array of 1-4 actionable suggestions

Focus on findings relevant to "{self.strategy.scan_focus}" and be specific about issues found."""

        prompt = f"""File: {file_path.name}
Type: {file_type}
Size: {len(content)} characters

Content:
{analysis_content}
{context_info}

Analyze this file:"""

        response = self.ollama.generate(prompt, system_prompt)

        # Parse AI response
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                analysis_data = json.loads(response[start:end])

                # Validate and clean the response
                return {
                    'relevance_score': max(0.0, min(1.0, float(analysis_data.get('relevance_score', 0.5)))),
                    'key_findings': analysis_data.get('key_findings', [])[:5],  # Limit to 5
                    'content_summary': analysis_data.get('content_summary', '')[:500],  # Limit length
                    'recommendations': analysis_data.get('recommendations', [])[:4]  # Limit to 4
                }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f'\033[1;33m[AI Parse Warning]:\033[0m Failed to parse response for {file_path.name}: {str(e)}')

        # Fallback analysis without AI
        return self._fallback_analysis(content, file_path)

    def _fallback_analysis(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Basic analysis when AI parsing fails"""
        relevance_score = self._calculate_keyword_relevance(content)

        # Basic content analysis
        lines = content.split('\n')
        word_count = len(content.split())

        key_findings = []
        if relevance_score > 0.3:
            key_findings.append(
                f"Contains {sum(1 for k in self.strategy.keywords if k.lower() in content.lower())} relevant keywords")

        if any(pattern.lower() in content.lower() for pattern in ['todo', 'fixme', 'hack', 'bug']):
            key_findings.append("Contains development notes or potential issues")

        if len(lines) > 500:
            key_findings.append("Large file - may need detailed review")

        return {
            'relevance_score': relevance_score,
            'key_findings': key_findings or [f"File contains {word_count} words in {len(lines)} lines"],
            'content_summary': f"{file_path.suffix} file with {word_count} words",
            'recommendations': ["Consider manual review based on file relevance"]
        }

    def _calculate_keyword_relevance(self, content: str) -> float:
        """Calculate relevance based on keyword matching"""
        if not self.strategy.keywords:
            return 0.5

        content_lower = content.lower()
        matches = sum(1 for keyword in self.strategy.keywords if keyword.lower() in content_lower)
        return min(matches / len(self.strategy.keywords), 1.0)

    def run_semantic_analysis(self, files: List[Path]) -> List[SemanticInsight]:
        """Run semantic analysis across all embedded files"""
        insights = []

        print(f'\033[1;36m[Semantic Analysis]:\033[0m Running {len(self.strategy.semantic_queries)} queries...')

        for query in self.strategy.semantic_queries:
            print(f'\033[1;90m  Query: "{query}"\033[0m')

            # Search across all embedded files
            matches = self.vector_db.semantic_search(query, top_k=15, similarity_threshold=0.5)

            if matches:
                insight = create_semantic_insight(query, matches)
                if insight:
                    insights.append(insight)
                    print(
                        f'\033[1;90m  Found {len(matches)} matches, {insight.high_confidence_matches} high confidence\033[0m')

        return insights

    def scan_directory(self, path: Path, extensions: List[str] = None) -> ScanResults:
        """Main scanning orchestrator"""
        scan_start_time = time.time()

        print(f'\033[1;33m[Scanning]:\033[0m Starting agentic analysis with {self.strategy.analysis_depth} depth...')

        # Find and prioritize files
        files = self.find_files(path, extensions)

        if not files:
            return ScanResults(
                goal=self.goal,
                session_id=self.session_id,
                strategy=self.strategy,
                total_files=0,
                analyzed_files=0,
                skipped_files=0,
                high_relevance_files=[],
                semantic_insights=[],
                summary="No files found matching criteria",
                recommendations=["Check file path and extensions", "Try different file extensions"],
                scan_timestamp=datetime.now().isoformat(),
                processing_time_seconds=time.time() - scan_start_time
            )

        # Limit files based on strategy
        max_files = min(self.strategy.max_files_to_analyze, len(files))
        files_to_analyze = files[:max_files]
        skipped_files = len(files) - len(files_to_analyze)

        print(f'\033[1;34m[Files]:\033[0m Found {len(files)}, analyzing top {len(files_to_analyze)}...')

        # Analyze files with AI
        analyses = []
        for i, file_path in enumerate(files_to_analyze):
            print(f'\033[1;36m[{i + 1}/{len(files_to_analyze)}]:\033[0m {file_path.name}', end='')

            analysis = self.analyze_file(file_path)
            if analysis:
                analyses.append(analysis)
                print(f' (score: {analysis.relevance_score:.2f})')
            else:
                print(' (failed)')

        # Run semantic analysis across all files
        semantic_insights = self.run_semantic_analysis(files_to_analyze)

        # Filter high relevance files
        high_relevance = [a for a in analyses if a.relevance_score > 0.6]
        high_relevance.sort(key=lambda x: x.relevance_score, reverse=True)

        # Generate AI-powered summary and recommendations
        summary = self._generate_summary(analyses, semantic_insights)
        recommendations = self._generate_recommendations(analyses, semantic_insights)

        # Store session in database
        self.vector_db.store_scan_session(
            self.session_id,
            self.goal,
            len(analyses),
            summary
        )

        processing_time = time.time() - scan_start_time

        return ScanResults(
            goal=self.goal,
            session_id=self.session_id,
            strategy=self.strategy,
            total_files=len(files),
            analyzed_files=len(analyses),
            skipped_files=skipped_files,
            high_relevance_files=high_relevance,
            semantic_insights=semantic_insights,
            summary=summary,
            recommendations=recommendations,
            scan_timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time
        )

    def _generate_summary(self, analyses: List[FileAnalysis], semantic_insights: List[SemanticInsight]) -> str:
        """Generate AI-powered summary of scan results"""
        if not analyses:
            return "No files were successfully analyzed."

        # Prepare summary data
        high_relevance_count = len([a for a in analyses if a.relevance_score > 0.6])
        avg_relevance = sum(a.relevance_score for a in analyses) / len(analyses) if analyses else 0

        top_findings = []
        for analysis in analyses[:5]:  # Top 5 files
            if analysis.key_findings:
                top_findings.extend(analysis.key_findings[:2])

        # Get file type distribution
        file_types = {}
        for analysis in analyses:
            ext = Path(analysis.file_path).suffix or 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1

        # Get top semantic insights
        high_confidence_insights = [s for s in semantic_insights if s.high_confidence_matches > 0]

        system_prompt = """You are an expert code and file analysis specialist. Create a concise, insightful 2-3 sentence summary of the scan results.

    Focus on:
    - Key discoveries related to the goal
    - Most important patterns found
    - Critical issues that need attention
    - Overall assessment of the codebase/files

    Be specific about findings and use technical language appropriate for developers."""

        prompt = f"""Scan Results Summary:

    Goal: "{self.goal}"
    Focus Area: {self.strategy.scan_focus}
    Analysis Depth: {self.strategy.analysis_depth}

    Files Analyzed: {len(analyses)} files
    High Relevance Files: {high_relevance_count} files (score > 0.6)
    Average Relevance Score: {avg_relevance:.2f}

    File Types Analyzed: {dict(list(file_types.items())[:5])}

    Top Key Findings:
    {chr(10).join(f"- {finding}" for finding in top_findings[:8])}

    Semantic Analysis:
    - Total Insights: {len(semantic_insights)}
    - High-Confidence Matches: {len(high_confidence_insights)}
    - Top Semantic Queries: {[s.query for s in semantic_insights[:3]]}

    Most Critical Patterns Found:
    {chr(10).join(f"- {pattern}" for pattern in self.strategy.patterns_to_find[:4])}

    Generate a professional, technical summary highlighting the most important discoveries and their implications:"""

        ai_summary = self.ollama.generate(prompt, system_prompt)

        # Clean up and validate AI response
        if ai_summary and len(ai_summary.strip()) > 20:
            # Remove any JSON formatting if present
            clean_summary = ai_summary.strip()
            if clean_summary.startswith('"') and clean_summary.endswith('"'):
                clean_summary = clean_summary[1:-1]

            # Ensure reasonable length (not too long or short)
            if len(clean_summary) > 500:
                sentences = clean_summary.split('. ')
                clean_summary = '. '.join(sentences[:3]) + '.'

            return clean_summary
        else:
            # Enhanced fallback summary with more context
            focus_desc = {
                'security': 'security vulnerabilities and risks',
                'bugs': 'potential bugs and code issues',
                'performance': 'performance bottlenecks and optimizations',
                'documentation': 'documentation gaps and issues',
                'structure': 'architectural and structural concerns',
                'general': 'code quality and improvement opportunities'
            }.get(self.strategy.scan_focus, 'issues and improvements')

            insights_desc = ""
            if semantic_insights:
                insights_desc = f" Vector analysis revealed {len(high_confidence_insights)} high-confidence semantic patterns across the codebase."

            top_file_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
            types_desc = f" Analyzed primarily {', '.join([f'{count} {ext}' for ext, count in top_file_types])} files."

            return f"Analyzed {len(analyses)} files focusing on {focus_desc}, with {high_relevance_count} files showing high relevance (avg score: {avg_relevance:.2f}).{insights_desc}{types_desc}"