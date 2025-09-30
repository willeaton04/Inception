
# Ollama API client for question answering and analysis


import os
import json
import time
from typing import Optional, Dict, Any, List
import logging
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaManager:
    """Manages Ollama API calls for question answering and file analysis"""

    def __init__(self,
                 model: str = "gemma2:2b",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """
        Initialize Ollama client

        Args:
            model: Model to use (e.g., "gemma2:2b")
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.client = ollama.Client(host='http://localhost:11434')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

        # Track usage
        self.total_tokens_used = 0

        self.logger.info(f"Ollama client initialized with model: {model}")

    def check_status(self) -> bool:
        """Check if Ollama API is accessible"""
        try:
            # Try a minimal API call
            self.client.list()
            return True
        except Exception as e:
            self.logger.error(f"Ollama API check failed: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def answer_question(self,
                        question: str,
                        context: str,
                        system_prompt: Optional[str] = None) -> str:
        """
        Answer a question based on provided context

        Args:
            question: User's question
            context: Relevant file content and information
            system_prompt: Optional custom system prompt

        Returns:
            Answer to the question
        """
        if not system_prompt:
            system_prompt = """You are an intelligent assistant that answers questions based on provided file content and context. 

Your task is to:
1. Carefully analyze the provided context from the user's files
2. Answer the user's question accurately based ONLY on the information available
3. If the answer is not in the provided context, clearly state that
4. Provide specific references to files when relevant
5. Be concise but thorough in your response

Important: Base your answer solely on the provided context. Do not make assumptions beyond what's explicitly stated in the files."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from files:\n\n{context}\n\nQuestion: {question}"}
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )

            # Track usage
            if response.get('total_duration'):
                self.total_tokens_used += response.get('eval_count', 0)

            answer = response['message']['content'].strip()
            self.logger.info(f"Generated answer for question: {question[:50]}...")

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return f"Error: Unable to generate answer - {str(e)}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyze the user's question to understand intent and extract key information

        Args:
            question: User's question

        Returns:
            Analysis of question intent and key terms
        """
        system_prompt = """You are an expert at analyzing questions to understand user intent.

Analyze the given question and return a JSON object with:
- intent: The primary intent (e.g., "explain", "find", "compare", "summarize", "debug", "analyze")
- key_concepts: List of important concepts/terms to search for
- scope: The scope of analysis needed ("specific", "broad", "comparative")
- file_types_relevant: List of file extensions that would be most relevant
- search_queries: List of 3-5 semantic search queries to find relevant content

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "num_predict": 500
                },
                format="json"
            )
            
            if response.get('total_duration'):
                self.total_tokens_used += response.get('eval_count', 0)

            analysis = json.loads(response['message']['content'])
            self.logger.info(f"Analyzed question intent: {analysis.get('intent', 'unknown')}")

            return analysis

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse intent analysis: {str(e)}")
            # Return default analysis
            return {
                "intent": "general",
                "key_concepts": question.split()[:5],
                "scope": "broad",
                "file_types_relevant": [],
                "search_queries": [question]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing question: {str(e)}")
            return {
                "intent": "error",
                "error": str(e),
                "key_concepts": [],
                "search_queries": [question]
            }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_file_insights(self,
                              file_content: str,
                              file_path: str,
                              question: str) -> Dict[str, Any]:
        """
        Extract relevant insights from a file based on the question

        Args:
            file_content: Content of the file
            file_path: Path to the file
            question: User's question for context

        Returns:
            Extracted insights relevant to the question
        """
        # Truncate content if too long
        max_content_length = 3000
        if len(file_content) > max_content_length:
            file_content = file_content[:max_content_length] + "\n... (truncated)"

        system_prompt = """Extract key insights from the file that are relevant to answering the user's question.

Return a JSON object with:
- relevance_score: 0.0-1.0 indicating how relevant this file is to the question
- key_points: List of important points relevant to the question
- direct_answers: Any direct answers to the question found in this file
- related_concepts: Related concepts that might help answer the question
- code_snippets: Any relevant code snippets (if applicable)

Be selective and only include information that helps answer the question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nFile: {file_path}\n\nContent:\n{file_content}"}
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "num_predict": 1000
                },
                format="json"
            )

            if response.get('total_duration'):
                self.total_tokens_used += response.get('eval_count', 0)

            insights = json.loads(response['message']['content'])
            self.logger.info(f"Extracted insights from {file_path}: relevance={insights.get('relevance_score', 0)}")

            return insights

        except Exception as e:
            self.logger.error(f"Error extracting insights from {file_path}: {str(e)}")
            return {
                "relevance_score": 0.0,
                "key_points": [],
                "direct_answers": [],
                "error": str(e)
            }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def synthesize_answer(self,
                          question: str,
                          file_insights: List[Dict[str, Any]],
                          semantic_context: Optional[str] = None) -> str:
        """
        Synthesize a comprehensive answer from multiple file insights

        Args:
            question: User's question
            file_insights: List of insights from analyzed files
            semantic_context: Additional context from semantic search

        Returns:
            Comprehensive answer
        """
        # Sort insights by relevance
        file_insights.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Build context from top insights
        context_parts = []
        for insight in file_insights[:10]:  # Limit to top 10 files
            if insight.get('relevance_score', 0) > 0.3:
                file_path = insight.get('file_path', 'unknown')
                context_parts.append(f"From {file_path}:")

                if insight.get('direct_answers'):
                    context_parts.append("Direct answers found:")
                    for answer in insight['direct_answers'][:3]:
                        context_parts.append(f"  - {answer}")

                if insight.get('key_points'):
                    context_parts.append("Key points:")
                    for point in insight['key_points'][:3]:
                        context_parts.append(f"  - {point}")

                context_parts.append("")

        if semantic_context:
            context_parts.append("Additional semantic context:")
            context_parts.append(semantic_context)

        context = "\n".join(context_parts)

        system_prompt = """You are synthesizing information from multiple files to provide a comprehensive answer.

Guidelines:
1. Provide a clear, direct answer to the question
2. Reference specific files when making claims
3. Structure your answer logically
4. If information is contradictory, acknowledge it
5. If the answer cannot be found in the provided context, state that clearly
6. Use examples from the files when helpful"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nInformation from files:\n{context}"}
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )

            if response.get('total_duration'):
                self.total_tokens_used += response.get('eval_count', 0)

            answer = response['message']['content'].strip()
            return answer

        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {str(e)}")
            return f"Error synthesizing answer: {str(e)}"

    def generate_follow_up_questions(self,
                                     question: str,
                                     answer: str,
                                     context: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the answer

        Args:
            question: Original question
            answer: Generated answer
            context: File context used

        Returns:
            List of follow-up questions
        """
        prompt = f"""Based on this Q&A exchange and file context, suggest 3-5 relevant follow-up questions that would help the user explore the topic deeper.

Original Question: {question}

Answer: {answer[:500]}...

Generate follow-up questions that:
1. Explore related aspects not fully covered
2. Dive deeper into specific points mentioned
3. Clarify any potential ambiguities
4. Connect to other relevant topics in the files

Return only the questions, one per line."""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 200
                }
            )

            if response.get('total_duration'):
                self.total_tokens_used += response.get('eval_count', 0)

            questions = response['message']['content'].strip().split('\n')
            # Clean and filter questions
            questions = [q.strip().lstrip('- ').lstrip('• ').strip()
                         for q in questions if q.strip() and '?' in q]

            return questions[:5]

        except Exception as e:
            self.logger.error(f"Error generating follow-up questions: {str(e)}")
            return []

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics"""
        return {
            "model": self.model,
            "total_tokens_used": self.total_tokens_used,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize client
    client = OllamaManager()

    # Test question analysis
    question = "How does the authentication system work in this codebase?"
    intent = client.analyze_question_intent(question)
    print(f"Question intent: {json.dumps(intent, indent=2)}")

    # Test answer generation
    context = """
    File: auth.py
    The authentication system uses JWT tokens for user sessions.
    Users log in with username and password, which are validated against the database.
    Upon successful authentication, a JWT token is generated with a 24-hour expiration.
    """

    answer = client.answer_question(question, context)
    print(f"\nAnswer: {answer}")

    # Show usage stats
    print(f"\nUsage stats: {client.get_usage_stats()}")
