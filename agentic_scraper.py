#!/usr/bin/env python3
"""
Agentic Question-Answering System
Analyzes files to answer user questions intelligently
"""

import json
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

# Handle imports with proper error messages
try:
    from ollama_client import OllamaManager
except ImportError:
    print("Error: ollama_client.py not found. Please ensure it's in the same directory.")
    raise

try:
    from vector_db import VectorDatabase
except ImportError:
    print("Error: vector_db.py not found. Please ensure it's in the same directory.")
    raise

try:
    from data_models import (
        FileAnalysis, SemanticInsight, ScanStrategy, ScanResults, ProcessingError,
        create_file_analysis, create_semantic_insight, create_default_strategy
    )
except ImportError:
    print("Error: data_models.py not found. Please ensure it's in the same directory.")
    raise


class AgenticQuestionAnswerer:
    """Orchestrates file analysis to answer user questions intelligently"""

    def __init__(self,
                 question: str,
                 ollama_model: str = 'gemma2:2b',
                 vector_db_path: str = "file_vectors.db"):
        """
        Initialize the question-answering system

        Args:
            question: User's question to answer
            ollama_model: Ollama model to use
            vector_db_path: Path for vector database storage
        """
        self.question = question
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)

        # Generate session ID
        self.session_id = hashlib.md5(f"{question}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Initialize components
        # print(f'\033[1;33m[Initializing]:\033[0m Setting up AI and vector database...')

        try:
            self.ollama = OllamaManager(model=ollama_model)
            self.vector_db = VectorDatabase(db_path=vector_db_path)
        except Exception as e:
            print(f'\033[1;31m[Error]:\033[0m Failed to initialize: {str(e)}')
            raise

        # Track processing
        self.processing_errors: List[ProcessingError] = []
        self.file_insights: List[Dict[str, Any]] = []

        print(f'\033[1;32m[Initialized]:\033[0m Session {self.session_id}')

        # Analyze question intent
        self.question_analysis = self._analyze_question()

        # Generate search strategy
        self.strategy = self._create_search_strategy()

        # For backward compatibility
        self.goal = self.question

    def _analyze_question(self) -> Dict[str, Any]:
        """Analyze the user's question to understand intent"""
        print(f'\033[1;36m[Analyzing Question]:\033[0m Understanding intent...')

        analysis = self.ollama.analyze_question_intent(self.question)

        print(f'\033[1;34m[Intent]:\033[0m {analysis.get("intent", "general")}')
        print(f'\033[1;34m[Scope]:\033[0m {analysis.get("scope", "broad")}')

        key_concepts = analysis.get("key_concepts", [])
        if key_concepts:
            print(f'\033[1;34m[Key Concepts]:\033[0m {", ".join(key_concepts[:5])}')

        return analysis

    def _create_search_strategy(self) -> ScanStrategy:
        """Create a search strategy based on question analysis"""

        # Extract components from question analysis
        key_concepts = self.question_analysis.get("key_concepts", [])
        file_types = self.question_analysis.get("file_types_relevant", [])
        search_queries = self.question_analysis.get("search_queries", [self.question])
        intent = self.question_analysis.get("intent", "general")

        # Map intent to scan focus
        intent_to_focus = {
            "explain": "documentation",
            "debug": "bugs",
            "find": "general",
            "compare": "structure",
            "analyze": "general",
            "summarize": "documentation"
        }
        scan_focus = intent_to_focus.get(intent, "general")

        # Determine analysis depth based on scope
        scope = self.question_analysis.get("scope", "broad")
        depth_map = {
            "specific": "deep",
            "broad": "medium",
            "comparative": "shallow"
        }
        analysis_depth = depth_map.get(scope, "medium")

        strategy = ScanStrategy(
            keywords=key_concepts,
            file_priorities=file_types if file_types else [".py", ".js", ".md", ".txt", ".java", ".cpp"],
            analysis_depth=analysis_depth,
            patterns_to_find=[],
            scan_focus=scan_focus,
            semantic_queries=search_queries,
            max_files_to_analyze=50  # Analyze more files for comprehensive answers
        )

        print(f'\033[1;34m[Strategy]:\033[0m Depth: {analysis_depth} | Focus: {scan_focus}')

        return strategy

    def find_relevant_files(self, path: Path, extensions: List[str] = None) -> List[Path]:
        """Find files relevant to answering the question"""
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
            except PermissionError as e:
                error = ProcessingError(
                    file_path=str(path),
                    error_type="PermissionError",
                    error_message=str(e)
                )
                self.processing_errors.append(error)
                print(f'\033[1;33m[Warning]:\033[0m Permission denied accessing some files')

        # Prioritize based on relevance to question
        return self._prioritize_files_for_question(files)

    def _should_include_file(self, file_path: Path, extensions: List[str]) -> bool:
        """Check if file should be included"""
        # Skip hidden files
        if any(part.startswith('.') for part in file_path.parts):
            return False

        # Skip common build/cache directories
        ignore_patterns = ['node_modules', '__pycache__', '.git', 'build', 'dist', 'target', 'bin', 'obj', 'venv']
        if any(ignore in str(file_path).lower() for ignore in ignore_patterns):
            return False

        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                return False
        except OSError:
            return False

        # Check extension if specified
        if extensions and file_path.suffix.lower() not in extensions:
            return False

        return True

    def _prioritize_files_for_question(self, files: List[Path]) -> List[Path]:
        """Prioritize files based on relevance to the question"""

        def relevance_score(file_path: Path) -> float:
            score = 0.0
            path_str = str(file_path).lower()

            # Check for key concepts in path
            for concept in self.strategy.keywords:
                if concept.lower() in path_str:
                    score += 10

            # Prioritize certain file types based on question intent
            intent = self.question_analysis.get("intent", "general")

            if intent == "explain" and file_path.suffix in ['.md', '.txt', '.rst', '.doc']:
                score += 15
            elif intent == "debug" and file_path.suffix in ['.py', '.js', '.java', '.log']:
                score += 15
            elif intent == "analyze" and file_path.suffix in ['.py', '.js', '.java', '.cpp']:
                score += 10

            # Prioritize main/index/readme files
            if any(name in file_path.name.lower() for name in ['readme', 'main', 'index', 'doc']):
                score += 8

            # Consider file size (prefer readable sizes)
            try:
                size = file_path.stat().st_size
                if 1024 < size < 100 * 1024:  # 1KB - 100KB
                    score += 5
            except OSError:
                pass

            return score

        return sorted(files, key=relevance_score, reverse=True)

    def analyze_file_for_question(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a file specifically for answering the question"""
        try:
            # Read file content
            content = self._read_file_safely(file_path)
            if not content:
                return None

            # Store in vector database for semantic search
            file_stats = file_path.stat()
            file_type = mimetypes.guess_type(str(file_path))[0] or "unknown"

            metadata = {
                'file_type': file_type,
                'size': file_stats.st_size,
                'modified': file_stats.st_mtime,
                'session_id': self.session_id,
                'question': self.question
            }

            # Store file in vector database
            file_id = self.vector_db.store_file(
                file_path=file_path,
                content=content,
                metadata=metadata
            )

            if not file_id:
                self.logger.warning(f"Failed to store {file_path} in vector database")

            # Extract insights relevant to the question
            insights = self.ollama.extract_file_insights(
                file_content=content,
                file_path=str(file_path),
                question=self.question
            )

            # Add file metadata to insights
            insights['file_path'] = str(file_path)
            insights['file_type'] = file_type
            insights['file_size'] = file_stats.st_size

            return insights

        except Exception as e:
            error = ProcessingError(
                file_path=str(file_path),
                error_type=type(e).__name__,
                error_message=str(e)
            )
            self.processing_errors.append(error)
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            return None

    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read file content"""
        try:
            # Check size
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                return ""
        except OSError:
            return ""

        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if len(content.strip()) < 10:  # Skip nearly empty files
                        return ""
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                break

        return ""

    def perform_semantic_search(self) -> List[Dict[str, Any]]:
        """Perform semantic search across all stored files"""
        semantic_results = []

        print(f'\033[1;36m[Semantic Search]:\033[0m Finding relevant content...')

        for query in self.strategy.semantic_queries:
            # Search for relevant chunks
            matches = self.vector_db.search_similar(
                query_text=query,
                limit=10,
                threshold=0.5
            )

            for match in matches:
                semantic_results.append({
                    'query': query,
                    'file_path': match['file_path'],
                    'content': match['content'],
                    'similarity': match['similarity'],
                    'metadata': match.get('metadata', {})
                })

        # Sort by similarity
        semantic_results.sort(key=lambda x: x['similarity'], reverse=True)

        if semantic_results:
            print(f'\033[1;90m  Found {len(semantic_results)} relevant chunks\033[0m')

        return semantic_results

    def answer_question(self, path: Path, extensions: List[str] = None) -> Dict[str, Any]:
        """
        Main method to answer the user's question based on files

        Args:
            path: Directory or file to analyze
            extensions: Optional list of file extensions to consider

        Returns:
            Dictionary containing answer and supporting information
        """
        start_time = time.time()

        print(f'\033[1;33m[Processing]:\033[0m Analyzing files to answer your question...')
        print(f'\033[1;34m[Question]:\033[0m {self.question}')

        # Find relevant files
        files = self.find_relevant_files(path, extensions)

        if not files:
            return {
                'status': 'error',
                'answer': 'No files found to analyze. Please check the path and try again.',
                'question': self.question,
                'processing_time': time.time() - start_time
            }

        # Limit number of files to analyze
        files_to_analyze = files[:self.strategy.max_files_to_analyze]

        print(f'\033[1;34m[Files]:\033[0m Found {len(files)}, analyzing top {len(files_to_analyze)}...')

        # Analyze each file for insights
        for i, file_path in enumerate(files_to_analyze):
            print(f'\033[1;36m[{i + 1}/{len(files_to_analyze)}]:\033[0m {file_path.name}', end='')

            insights = self.analyze_file_for_question(file_path)
            if insights:
                self.file_insights.append(insights)
                relevance = insights.get('relevance_score', 0)
                print(f' (relevance: {relevance:.2f})')
            else:
                print(' (skipped)')

        # Perform semantic search
        semantic_results = self.perform_semantic_search()

        # Build semantic context
        semantic_context = self._build_semantic_context(semantic_results)

        # Filter highly relevant insights
        relevant_insights = [
            insight for insight in self.file_insights
            if insight.get('relevance_score', 0) > 0.3
        ]

        if not relevant_insights and not semantic_results:
            return {
                'status': 'no_relevant_content',
                'answer': 'I couldn\'t find relevant information in the analyzed files to answer your question. The files may not contain information related to your query.',
                'question': self.question,
                'files_analyzed': len(self.file_insights),
                'processing_time': time.time() - start_time
            }

        # Synthesize final answer
        print(f'\033[1;33m[Synthesizing]:\033[0m Generating comprehensive answer...')

        answer = self.ollama.synthesize_answer(
            question=self.question,
            file_insights=relevant_insights,
            semantic_context=semantic_context
        )

        # Generate follow-up questions
        follow_up_questions = self.ollama.generate_follow_up_questions(
            question=self.question,
            answer=answer,
            context=semantic_context[:1000] if semantic_context else ""
        )

        # Prepare detailed results
        results = {
            'status': 'success',
            'question': self.question,
            'answer': answer,
            'confidence': self._calculate_confidence(relevant_insights),
            'supporting_files': self._get_supporting_files(relevant_insights),
            'key_insights': self._extract_key_insights(relevant_insights),
            'follow_up_questions': follow_up_questions,
            'statistics': {
                'files_found': len(files),
                'files_analyzed': len(self.file_insights),
                'relevant_files': len(relevant_insights),
                'semantic_matches': len(semantic_results),
                'processing_time_seconds': time.time() - start_time,
                'session_id': self.session_id
            },
            'ollama_usage': self.ollama.get_usage_stats()
        }

        # Print summary
        self._print_summary(results)

        return results

    def _build_semantic_context(self, semantic_results: List[Dict[str, Any]]) -> str:
        """Build context from semantic search results"""
        if not semantic_results:
            return ""

        context_parts = []
        seen_content = set()

        for result in semantic_results[:15]:  # Top 15 matches
            content_hash = hashlib.md5(result['content'].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                context_parts.append(f"From {result['file_path']}:")
                context_parts.append(result['content'][:300])
                context_parts.append("")

        return "\n".join(context_parts)

    def _calculate_confidence(self, relevant_insights: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        if not relevant_insights:
            return 0.0

        # Average relevance score of top insights
        top_scores = sorted(
            [i.get('relevance_score', 0) for i in relevant_insights],
            reverse=True
        )[:5]

        if top_scores:
            avg_score = sum(top_scores) / len(top_scores)
            # Boost confidence if we have direct answers
            has_direct_answers = any(
                i.get('direct_answers') for i in relevant_insights
            )
            if has_direct_answers:
                avg_score = min(1.0, avg_score * 1.2)
            return round(avg_score, 2)

        return 0.0

    def _get_supporting_files(self, relevant_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get list of supporting files with their relevance"""
        supporting_files = []

        for insight in relevant_insights:
            if insight.get('relevance_score', 0) > 0.3:
                supporting_files.append({
                    'path': insight.get('file_path', 'unknown'),
                    'relevance': insight.get('relevance_score', 0),
                    'type': insight.get('file_type', 'unknown'),
                    'has_direct_answer': bool(insight.get('direct_answers'))
                })

        # Sort by relevance
        supporting_files.sort(key=lambda x: x['relevance'], reverse=True)

        return supporting_files[:10]  # Top 10 files

    def _extract_key_insights(self, relevant_insights: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from all relevant files"""
        all_insights = []

        for insight in relevant_insights:
            # Add direct answers
            for answer in insight.get('direct_answers', [])[:2]:
                all_insights.append(f"[Direct] {answer}")

            # Add key points
            for point in insight.get('key_points', [])[:2]:
                all_insights.append(f"[Finding] {point}")

        # Deduplicate similar insights
        unique_insights = []
        seen = set()

        for insight in all_insights:
            # Simple deduplication based on first 50 chars
            key = insight[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_insights.append(insight)

        return unique_insights[:10]  # Top 10 insights

    def _print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results"""
        print('\n' + '=' * 60)
        print('\033[1;32m[Answer Generated Successfully]\033[0m')
        print('=' * 60)

        print(f'\n\033[1;34m[Question]:\033[0m {results["question"]}')
        print(f'\n\033[1;32m[Answer]:\033[0m\n{results["answer"][:500]}...'
              if len(results["answer"]) > 500 else f'\n\033[1;32m[Answer]:\033[0m\n{results["answer"]}')

        print(f'\n\033[1;34m[Confidence]:\033[0m {results["confidence"] * 100:.0f}%')

        stats = results['statistics']
        print(f'\n\033[1;34m[Statistics]:\033[0m')
        print(f'  • Files analyzed: {stats["files_analyzed"]}')
        print(f'  • Relevant files: {stats["relevant_files"]}')
        print(f'  • Processing time: {stats["processing_time_seconds"]:.2f}s')

        if results.get('follow_up_questions'):
            print(f'\n\033[1;34m[Suggested Follow-up Questions]:\033[0m')
            for i, q in enumerate(results['follow_up_questions'][:3], 1):
                print(f'  {i}. {q}')

        usage = results.get('ollama_usage', {})
        if usage:
            print(f'\n\033[1;34m[API Usage]:\033[0m')
            print(f'  • Model: {usage.get("model")}')
            print(f'  • Tokens: {usage.get("total_tokens_used")}')

        print('=' * 60)

    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export results to a file"""
        output_file = Path(output_path)

        # Add timestamp to results
        results['export_timestamp'] = datetime.now().isoformat()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f'\n\033[1;32m[Exported]:\033[0m Results saved to {output_file}')
        except Exception as e:
            print(f'\n\033[1;31m[Export Error]:\033[0m {str(e)}')


# Maintain backward compatibility with old name
class AgenticFileScraper(AgenticQuestionAnswerer):
    """Backward compatibility wrapper"""

    def __init__(self, goal: str, ollama_model: str = 'gemma2:2b', vector_db_path: str = "file_vectors.db"):
        # Convert old parameters to new ones
        super().__init__(
            question=goal,
            ollama_model=ollama_model,
            vector_db_path=vector_db_path
        )

    def scan_directory(self, path: Path, extensions: List[str] = None) -> Any:
        """Backward compatibility method"""
        return self.answer_question(path, extensions)
