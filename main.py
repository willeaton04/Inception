#!/usr/bin/env python3
"""
Main entry point for the Agentic Question-Answering System
Analyzes files to answer user questions using OpenAI and vector database
"""

import sys
import os

# Import our modular components
from cli import (
    parse_sys_argv, validate_arguments, validate_filepath,
    parse_extensions, display_help, display_parsed_args
)


def main():
    """Main entry point for the agentic question answering system"""
    try:
        # Check if any arguments were provided
        if len(sys.argv) == 1:
            display_help()
            sys.exit(1)

        # Parse arguments using existing functions
        parsed_args = parse_sys_argv()

        # Check for --goal (backward compatibility) and convert to --question
        if parsed_args.get('goal') and not parsed_args.get('question'):
            parsed_args['question'] = parsed_args['goal']

        # Also check for 'goal' in sys.argv for backward compatibility
        for i, arg in enumerate(sys.argv):
            if arg == '--goal' and i + 1 < len(sys.argv):
                if not parsed_args.get('question'):
                    parsed_args['question'] = sys.argv[i + 1]

        # Validate arguments (note: validate_arguments expects 'question' not 'goal')
        if not validate_arguments(parsed_args):
            display_help()
            sys.exit(1)

        # Validate filepath
        path = validate_filepath(parsed_args['filepath'])

        # Parse extensions
        extensions = parse_extensions(parsed_args.get('ext'))

        # Display parsed arguments
        display_parsed_args(parsed_args, path, extensions)

        # Initialize the question-answering system with error handling
        print('\033[1;33m[Initializing]:\033[0m Setting up AI and vector database...')

        try:
            # Import the correct class
            from agentic_scraper import AgenticQuestionAnswerer

            # Create the QA system with proper parameters
            qa_system = AgenticQuestionAnswerer(
                question=parsed_args['question'],  # Use 'question' not 'goal'
                openai_api_key=parsed_args.get('api_key'),
                openai_model=parsed_args.get('model', 'gpt-3.5-turbo'),  # Default to gpt-3.5-turbo
                vector_db_path="file_vectors.db"
            )

        except ImportError as e:
            print(f'\033[1;31m[Import Error]:\033[0m {str(e)}')
            print('\033[1;33m[Help]:\033[0m Make sure all required files are present:')
            print('  - agentic_scraper.py')
            print('  - openai_client.py (or ollama_client.py if using OpenAI wrapper)')
            print('  - vector_db.py')
            print('  - data_models.py')
            sys.exit(1)
        except ValueError as e:
            # This catches the OpenAI API key error
            print(f'\033[1;31m[Configuration Error]:\033[0m {str(e)}')
            print('\033[1;33m[Help]:\033[0m Set your OpenAI API key:')
            print('  export OPENAI_API_KEY="your-api-key-here"')
            print('  or use --api-key parameter')
            sys.exit(1)
        except Exception as e:
            print(f'\033[1;31m[Initialization Error]:\033[0m {str(e)}')
            sys.exit(1)

        # Run the question-answering analysis
        try:
            results = qa_system.answer_question(path, extensions)

            # Check if we got valid results
            if not results:
                print(f'\033[1;31m[Error]:\033[0m No results returned')
                sys.exit(1)

            # Display results based on status
            if results.get('status') == 'success':
                # Results are already printed by the qa_system

                # Export results if requested
                if parsed_args.get('output'):
                    qa_system.export_results(results, parsed_args['output'])

            elif results.get('status') == 'no_relevant_content':
                print(f'\033[1;33m[No Results]:\033[0m {results.get("answer", "No relevant content found")}')

            elif results.get('status') == 'error':
                print(f'\033[1;31m[Error]:\033[0m {results.get("answer", "Unknown error occurred")}')
                sys.exit(1)

        except Exception as e:
            print(f'\033[1;31m[Processing Error]:\033[0m {str(e)}')

            # Try to get processing stats even if analysis failed
            try:
                stats = qa_system.get_processing_stats() if hasattr(qa_system, 'get_processing_stats') else {}
                if stats and stats.get('processing_errors'):
                    print(f'\033[1;33m[Processing Errors]:\033[0m {len(stats["processing_errors"])} errors encountered')
                    for error in stats['processing_errors'][:3]:  # Show first 3 errors
                        print(f'  - {error.get("error_type", "Unknown")}: {error.get("file_path", "Unknown file")}')
            except:
                pass

            sys.exit(1)

        # Show additional statistics if available
        try:
            if hasattr(qa_system, 'openai'):
                usage_stats = qa_system.openai.get_usage_stats()
                if usage_stats:
                    print(f'\n\033[1;34m[OpenAI Usage]:\033[0m')
                    print(f'  Tokens: {usage_stats.get("total_tokens_used", 0)}')
                    print(f'  Estimated cost: {usage_stats.get("estimated_cost", "$0.00")}')
        except:
            pass

    except KeyboardInterrupt:
        print('\n\033[1;31m[Interrupted]:\033[0m Process cancelled by user')
        sys.exit(130)
    except Exception as e:
        print(f'\033[1;31m[Unexpected Error]:\033[0m {str(e)}')
        print('\033[1;33m[Debug Info]:\033[0m Run with Python in debug mode for more details')
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_requirements():
    """Print installation requirements"""
    print("\033[1;32m[Agentic Question-Answering System Requirements]\033[0m")
    print()
    print("Python packages:")
    print("  pip install openai chromadb sentence-transformers numpy")
    print()
    print("OpenAI API Setup:")
    print("  1. Get API key from https://platform.openai.com/api-keys")
    print("  2. Set environment variable:")
    print("     export OPENAI_API_KEY='your-api-key-here'")
    print()
    print("Models available:")
    print("  - gpt-4-turbo-preview (best quality)")
    print("  - gpt-4 (high quality)")
    print("  - gpt-3.5-turbo (fast and affordable)")
    print()


def print_version():
    """Print version information"""
    print("Agentic Question-Answering System v2.0.0")
    print("AI-powered file analysis using OpenAI and ChromaDB")
    print("Answers questions based on your codebase and documents")


if __name__ == "__main__":
    # Handle special arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--requirements', '--deps']:
            print_requirements()
            sys.exit(0)
        elif sys.argv[1] in ['--version', '-v']:
            print_version()
            sys.exit(0)

    main()