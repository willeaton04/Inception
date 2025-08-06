#!/usr/bin/env python3
"""
Main entry point for the Agentic File Scraper
Combines CLI parsing, Ollama AI, and vector database for intelligent file analysis
"""

import sys
import time
from pathlib import Path

# Import our modular components
from cli.cli import (
    parse_sys_argv, validate_arguments, validate_filepath,
    parse_extensions, display_help, display_parsed_args
)
from file_scraper.agentic_scraper import AgenticFileScraper
from summarizer.result_formatter import format_results


def main():
    """Main entry point for the agentic file scraper"""
    try:
        # Check if any arguments were provided
        if len(sys.argv) == 1:
            display_help()
            sys.exit(1)

        # Parse arguments using existing functions
        parsed_args = parse_sys_argv()

        # Validate arguments
        if not validate_arguments(parsed_args):
            display_help()
            sys.exit(1)

        # Validate filepath
        path = validate_filepath(parsed_args['filepath'])

        # Parse extensions
        extensions = parse_extensions(parsed_args['ext'])

        # Display parsed arguments
        display_parsed_args(parsed_args, path, extensions)

        # Initialize agentic scraper with error handling
        print('\033[1;33m[Initializing]:\033[0m Setting up AI and vector database...')

        try:
            scraper = AgenticFileScraper(
                goal=parsed_args['goal'],
                ollama_model='phi3',  # Can be made configurable
                vector_db_path="file_vectors.db"  # Can be made configurable
            )
        except RuntimeError as e:
            print(f'\033[1;31m[Initialization Error]:\033[0m {str(e)}')
            print('\033[1;33m[Help]:\033[0m Make sure Ollama is installed and running')
            print('\033[1;33m[Help]:\033[0m Install: curl -fsSL https://ollama.ai/install.sh | sh')
            print('\033[1;33m[Help]:\033[0m Start: ollama serve')
            sys.exit(1)

        # Run the intelligent scan
        try:
            results = scraper.scan_directory(path, extensions)
        except Exception as e:
            print(f'\033[1;31m[Scan Error]:\033[0m {str(e)}')

            # Try to get processing stats even if scan failed
            try:
                stats = scraper.get_processing_stats()
                if stats['processing_errors']:
                    print(f'\033[1;33m[Processing Errors]:\033[0m {len(stats["processing_errors"])} errors encountered')
                    for error in stats['processing_errors'][:3]:  # Show first 3 errors
                        print(f'  - {error["error_type"]}: {error["file_path"]}')
            except:
                pass

            sys.exit(1)

        # Display comprehensive results
        format_results(results, parsed_args['option'])

        # Show additional statistics if verbose
        stats = scraper.get_processing_stats()
        if stats['processing_errors']:
            print(f'\033[1;33m[Processing Warnings]:\033[0m {len(stats["processing_errors"])} files had issues')

        # Show vector database stats
        vector_stats = stats.get('vector_db_stats', {})
        if vector_stats:
            print(f'\033[1;34m[Vector DB]:\033[0m {vector_stats.get("total_embeddings", 0)} embeddings, '
                  f'{vector_stats.get("unique_files", 0)} unique files')

    except KeyboardInterrupt:
        print('\n\033[1;31m[Interrupted]:\033[0m Process cancelled by user')
        sys.exit(1)
    except Exception as e:
        print(f'\033[1;31m[Unexpected Error]:\033[0m {str(e)}')
        print('\033[1;33m[Debug Info]:\033[0m Run with Python in debug mode for more details')
        sys.exit(1)


def print_requirements():
    """Print installation requirements"""
    print("\033[1;32m[Agentic File Scraper Requirements]\033[0m")
    print()
    print("Python packages:")
    print("  pip install sentence-transformers numpy requests")
    print()
    print("Ollama installation:")
    print("  curl -fsSL https://ollama.ai/install.sh | sh")
    print("  ollama serve")
    print("  ollama pull phi3")
    print()


def print_version():
    """Print version information"""
    print("Agentic File Scraper v1.0.0")
    print("AI-powered file analysis with vector embeddings")


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