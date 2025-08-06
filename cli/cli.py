#!/usr/bin/env python3
"""
CLI argument parsing and display utilities
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_sys_argv() -> Optional[Dict[str, str]]:
    """Parse command line arguments using sys.argv"""
    args = sys.argv[1:]  # Skip script name

    if not args:
        return None

    parsed = {
        'filepath': None,
        'goal': None,
        'ext': None,
        'option': None
    }

    # First argument should be filepath
    if args and not args[0].startswith('--'):
        parsed['filepath'] = args[0]
        args = args[1:]

    # Parse remaining arguments
    i = 0
    while i < len(args):
        arg = args[i]

        if arg == '--goal':
            if i + 1 < len(args):
                parsed['goal'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --goal requires a value\033[0m')
                sys.exit(1)

        elif arg == '--ext':
            if i + 1 < len(args):
                parsed['ext'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --ext requires a value\033[0m')
                sys.exit(1)

        elif arg == '--option':
            if i + 1 < len(args):
                parsed['option'] = args[i + 1]
                i += 2
            else:
                print('\033[1;31m[Error]: --option requires a value\033[0m')
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

    if not parsed_args['goal']:
        print('\033[1;31m[Error]: --goal is required\033[0m')
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
    print('\033[1;32m[Agentic File Scraper Help]\033[0m')
    print()
    print(
        '\033[1;31m[Usage]: python scraper.py <filepath> --goal "your goal" [--ext extensions] [--option value]\033[0m')
    print()
    print("\033[1m[Required Arguments]:")
    print("   <filepath>           Path to folder or file to process")
    print("   --goal 'text'        Objective for the scraping process\033[0m")
    print()
    print("\033[1m[Optional Arguments]:")
    print("   --ext py,js,txt      Comma-separated file extensions to include")
    print("   --option value       Output file path or processing mode")
    print("   --help, -h           Show this help message\033[0m")
    print()
    print("\033[1m[Examples]:")
    print("   python scraper.py ./src --goal 'find security vulnerabilities'")
    print("   python scraper.py ./project --goal 'analyze code quality' --ext py,js")
    print("   python scraper.py ./docs --goal 'extract key concepts' --option report.json")
    print("   python scraper.py --help\033[0m")


def display_parsed_args(args: Dict[str, str], path: Path, extensions: List[str]) -> None:
    """Display the parsed arguments in a formatted way"""
    print('\033[1;32m[Agentic File Scraper Started]\033[0m')
    print(f'\033[1;34m[Filepath]:\033[0m {path.absolute()}')
    print(f'\033[1;34m[Type]:\033[0m {"Directory" if path.is_dir() else "File"}')
    print(f'\033[1;34m[Goal]:\033[0m {args["goal"]}')

    if extensions:
        print(f'\033[1;34m[Extensions]:\033[0m {", ".join(extensions)}')
    else:
        print('\033[1;34m[Extensions]:\033[0m All file types')

    if args['option']:
        print(f'\033[1;34m[Options]:\033[0m {args["option"]}')

    print('-' * 50)