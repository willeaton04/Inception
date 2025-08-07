#!/usr/bin/env python3
"""
CLI for Agentic Question-Answering System
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional


def parse_sys_argv() -> Optional[Dict[str, str]]:
    """Parse command line arguments using sys.argv"""
    args = sys.argv[1:]  # Skip script name

    if not args:
        return None

    parsed = {
        'filepath': None,
        'question': None,
        'ext': None,
        'output': None,
        'model': None,
        'api_key': None
    }

    # First argument should be filepath
    if args and not args[0].startswith('--'):
        parsed['filepath'] = args[0]
        args = args[1:]

    # Parse remaining arguments
    i = 0
    while i < len(args):
        arg = args[i]

        if arg == '--question' or arg == '-q':
            if i + 1 < len(args):
                parsed['question'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --question requires a value\033[0m')
                sys.exit(1)

        elif arg == '--ext' or arg == '-e':
            if i + 1 < len(args):
                parsed['ext'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --ext requires a value\033[0m')
                sys.exit(1)

        elif arg == '--output' or arg == '-o':
            if i + 1 < len(args):
                parsed['output'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --output requires a value\033[0m')
                sys.exit(1)

        elif arg == '--model' or arg == '-m':
            if i + 1 < len(args):
                parsed['model'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --model requires a value\033[0m')
                sys.exit(1)

        elif arg == '--api-key':
            if i + 1 < len(args):
                parsed['api_key'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --api-key requires a value\033[0m')
                sys.exit(1)

        elif arg == '--help' or arg == '-h':
            display_help()
            sys.exit(0)

        elif arg.startswith('--'):
            print(f'\033[1;31m[Error]: Unknown argument: {arg}\033[0m')
            sys.exit(1)

        else:
            print(f'\033[1;31m[Error]: Unexpected argument: {arg}\033[0m')
            sys.exit(1)

    return parsed


def validate_arguments(parsed_args: Dict[str, str]) -> bool:
    """Validate required arguments"""
    if not parsed_args:
        return False

    if not parsed_args['filepath']:
        print('\033[1;31m[Error]: Filepath is required\033[0m')
        return False

    if not parsed_args['question']:
        print('\033[1;31m[Error]: --question is required\033[0m')
        return False

    # Check for API key
    if not parsed_args['api_key'] and not os.getenv('OPENAI_API_KEY'):
        print(
            '\033[1;31m[Error]: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key\033[0m')
        return False

    return True


def validate_filepath(filepath: str) -> Path:
    """Validate that the provided filepath exists"""
    path = Path(filepath)
    if not path.exists():
        print(f'\033[1;31m[Error]: Path "{filepath}" does not exist\033[0m')
        sys.exit(1)
    return path


def parse_extensions(ext_string: Optional[str]) -> List[str]:
    """Parse comma-separated extensions into a list"""
    if not ext_string:
        return []

    extensions = [ext.strip().lower() for ext in ext_string.split(',')]
    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    return extensions


def display_help() -> None:
    """Display usage help"""
    print('\033[1;32m╔════════════════════════════════════════════════════════════════╗\033[0m')
    print('\033[1;32m║         Agentic Question-Answering System Help                 ║\033[0m')
    print('\033[1;32m╚════════════════════════════════════════════════════════════════╝\033[0m')
    print()
    print('\033[1;33m[Usage]:\033[0m')
    print('  python main.py <filepath> --question "your question" [options]')
    print()
    print('\033[1;36m[Required Arguments]:\033[0m')
    print('  \033[1m<filepath>\033[0m              Path to folder or file to analyze')
    print('  \033[1m--question, -q\033[0m          Question you want answered based on the files')
    print()
    print('\033[1;36m[Optional Arguments]:\033[0m')
    print('  \033[1m--ext, -e\033[0m               Comma-separated file extensions to include')
    print('                          (e.g., "py,js,md" or ".py,.js,.md")')
    print('  \033[1m--output, -o\033[0m            Path to save results as JSON')
    print('  \033[1m--model, -m\033[0m             OpenAI model to use')
    print('                          Options: gpt-4-turbo-preview (default), gpt-4, gpt-3.5-turbo')
    print('  \033[1m--api-key\033[0m               OpenAI API key (or set OPENAI_API_KEY env var)')
    print('  \033[1m--help, -h\033[0m              Show this help message')
    print()
    print('\033[1;36m[Examples]:\033[0m')
    print()
    print('  \033[1;90m# Ask about authentication in a codebase\033[0m')
    print('  python qa_system.py ./src --question "How does the authentication system work?"')
    print()
    print('  \033[1;90m# Ask about specific file types\033[0m')
    print('  python qa_system.py ./project --question "What design patterns are used?" --ext py,js')
    print()
    print('  \033[1;90m# Save results to file\033[0m')
    print('  python qa_system.py ./docs --question "What are the main features?" --output results.json')
    print()
    print('  \033[1;90m# Use specific model\033[0m')
    print('  python qa_system.py ./code --question "Find security issues" --model gpt-4')
    print()
    print('\033[1;36m[Question Examples]:\033[0m')
    print('  • "How does the authentication system work?"')
    print('  • "What database models are defined in this project?"')
    print('  • "Explain the main architecture of this application"')
    print('  • "What external APIs does this code interact with?"')
    print('  • "Find and explain any security vulnerabilities"')
    print('  • "What are the main features of this application?"')
    print('  • "How is error handling implemented?"')
    print('  • "What design patterns are used in this codebase?"')
    print()
    print('\033[1;33m[Notes]:\033[0m')
    print('  • The system analyzes files to provide comprehensive answers')
    print('  • It uses semantic search to find relevant content')
    print('  • Larger codebases may take longer to process')
    print('  • API costs apply based on token usage')
    print()
    print('\033[1;32m════════════════════════════════════════════════════════════════\033[0m')


def display_parsed_args(args: Dict[str, str], path: Path, extensions: List[str]) -> None:
    """Display the parsed arguments in a formatted way"""
    print('\033[1;32m╔════════════════════════════════════════════════════════════════╗\033[0m')
    print('\033[1;32m║         Agentic Question-Answering System Started             ║\033[0m')
    print('\033[1;32m╚════════════════════════════════════════════════════════════════╝\033[0m')
    print()
    print(f'\033[1;34m[Path]:\033[0m {path.absolute()}')
    print(f'\033[1;34m[Type]:\033[0m {"Directory" if path.is_dir() else "File"}')

    # Display question (truncate if too long)
    question = args["question"]
    if len(question) > 80:
        question_display = question[:77] + "..."
    else:
        question_display = question
    print(f'\033[1;34m[Question]:\033[0m {question_display}')

    if extensions:
        print(f'\033[1;34m[Extensions]:\033[0m {", ".join(extensions)}')
    else:
        print('\033[1;34m[Extensions]:\033[0m All supported file types')

    if args.get('output'):
        print(f'\033[1;34m[Output]:\033[0m Results will be saved to {args["output"]}')

    if args.get('model'):
        print(f'\033[1;34m[Model]:\033[0m {args["model"]}')
    else:
        print('\033[1;34m[Model]:\033[0m gpt-4-turbo-preview (default)')

    print('\n' + '─' * 60)
    print()


def display_welcome_banner():
    """Display a welcome banner"""
    print()
    print('\033[1;36m' + '=' * 60 + '\033[0m')
    print('\033[1;36m' + ' ' * 15 + 'AGENTIC QUESTION-ANSWERING SYSTEM' + ' ' * 11 + '\033[0m')
    print('\033[1;36m' + ' ' * 18 + 'Powered by OpenAI & ChromaDB' + ' ' * 14 + '\033[0m')
    print('\033[1;36m' + '=' * 60 + '\033[0m')
    print()


def main():
    """Main CLI entry point"""
    # Parse arguments
    parsed_args = parse_sys_argv()

    # Show help if no arguments
    if not parsed_args or not parsed_args['filepath']:
        display_welcome_banner()
        display_help()
        sys.exit(0)

    # Validate arguments
    if not validate_arguments(parsed_args):
        print('\nRun with --help for usage information')
        sys.exit(1)

    # Validate filepath
    path = validate_filepath(parsed_args['filepath'])

    # Parse extensions
    extensions = parse_extensions(parsed_args['ext'])

    # Display configuration
    display_parsed_args(parsed_args, path, extensions)

    # Import the main system
    try:
        from agentic_scraper import AgenticQuestionAnswerer
    except ImportError as e:
        print(f'\033[1;31m[Error]: Failed to import system components: {str(e)}\033[0m')
        print('\nPlease ensure all dependencies are installed:')
        print('  pip install openai chromadb sentence-transformers')
        sys.exit(1)

    # Initialize the system
    try:
        qa_system = AgenticQuestionAnswerer(
            question=parsed_args['question'],
            openai_api_key=parsed_args.get('api_key'),
            openai_model=parsed_args.get('model', 'gpt-4-turbo-preview')
        )
    except Exception as e:
        print(f'\033[1;31m[Error]: Failed to initialize system: {str(e)}\033[0m')
        sys.exit(1)

    # Process the question
    try:
        results = qa_system.answer_question(path, extensions)

        # Export results if requested
        if parsed_args.get('output'):
            qa_system.export_results(results, parsed_args['output'])

        # Return success
        sys.exit(0 if results['status'] == 'success' else 1)

    except KeyboardInterrupt:
        print('\n\033[1;33m[Interrupted]: Process cancelled by user\033[0m')
        sys.exit(130)
    except Exception as e:
        print(f'\n\033[1;31m[Error]: {str(e)}\033[0m')
        sys.exit(1)


if __name__ == "__main__":
    main()