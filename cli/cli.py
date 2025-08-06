import sys
from pathlib import Path


def parse_sys_argv():
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

        elif arg.startswith('--'):
            print(f'\033[1;31m[Error]: Unknown argument: {arg}\033[0m')
            sys.exit(1)

        else:
            print(f'\033[1;31m[Error]: Unexpected argument: {arg}\033[0m')
            sys.exit(1)

    return parsed


def validate_arguments(parsed_args):
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


def validate_filepath(filepath):
    """Validate that the provided filepath exists"""
    path = Path(filepath)
    if not path.exists():
        print(f'\033[1;31m[Error]: Path "{filepath}" does not exist\033[0m')
        sys.exit(1)
    return path


def parse_extensions(ext_string):
    """Parse comma-separated extensions into a list"""
    if not ext_string:
        return []

    extensions = [ext.strip().lower() for ext in ext_string.split(',')]
    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    return extensions


def display_help():
    """Display usage help"""
    print(
        '\033[1;31m[Usage]: python cli/cli.py <filepath> --goal "your goal" [--ext extensions] [--option value]\033[0m')
    print("\033[1;31m[Error]: Run inception with required arguments\033[0m")
    print("\033[1;32m[Run]: python cli/cli.py <filepath> --goal 'your goal'\033[0m")
    print("\033[1m[Args]:")
    print("   - Filepath: <folder/path> | Path to folder or file to process")
    print("   - Goal: --goal 'your goal' | Objective for the inception process")
    print("   - Extensions: --ext py,js,txt | Optional file types to include")
    print("   - Options: --option value | Optional output file or processing mode\033[0m")
    print()
    print("\033[1m[Examples]:")
    print("   python cli/cli.py ./src --goal 'analyze code quality'")
    print("   python cli/cli.py ./project --goal 'find bugs' --ext py,js,ts")
    print("   python cli/cli.py ./docs --goal 'extract summaries' --ext md,txt --option output.txt\033[0m")


def display_parsed_args(args, path, extensions):
    """Display the parsed arguments in a formatted way"""
    print('\033[1;32m[Inception CLI Started]\033[0m')
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


def find_files(path, extensions):
    """Find files matching the criteria"""
    files = []

    if path.is_file():
        if not extensions or path.suffix.lower() in extensions:
            files.append(path)
    else:
        # Directory - recursively find files
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    if not extensions or file_path.suffix.lower() in extensions:
                        files.append(file_path)
        except PermissionError as e:
            print(f'\033[1;33m[Warning]: Permission denied accessing some files in {path}\033[0m')
        except Exception as e:
            print(f'\033[1;31m[Error]: Error scanning directory {path}: {str(e)}\033[0m')

    return files


def process_files(files, goal, option=None):
    """Process the found files according to the goal"""
    print(f'\033[1;33m[Processing {len(files)} file(s)]:\033[0m')

    # Display files to be processed
    for i, file_path in enumerate(files):
        if i < 10:  # Show first 10 files
            relative_path = file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path
            print(f'  - {relative_path}')
        elif i == 10:
            print(f'  ... and {len(files) - 10} more files')
            break

    print()
    print(f'\033[1;35m[Goal]:\033[0m {goal}')

    # Handle different option types
    if option:
        if option.lower().endswith(('.txt', '.md', '.json', '.csv')):
            print(f'\033[1;36m[Output File]:\033[0m Results will be written to {option}')
        elif option.lower() in ['verbose', 'debug', 'quiet']:
            print(f'\033[1;36m[Mode]:\033[0m Processing in {option.lower()} mode')
        else:
            print(f'\033[1;36m[Option]:\033[0m Processing with custom option: {option}')

    print('\033[1;32m[Ready]:\033[0m File parsing ready - implement your processing logic here')

    # The files list and parsed arguments are ready for your inception processing

    # Commented out example processing:
    # --------------------------------------------------
    # print('\033[1;34m[Status]:\033[0m Analyzing files...')
    #
    # # Handle file output
    # if option and option.lower().endswith(('.txt', '.md', '.json', '.csv')):
    #     try:
    #         output_path = Path(option)
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         with open(output_path, 'w', encoding='utf-8') as f:
    #             f.write(f"Inception Results\n")
    #             f.write(f"Goal: {goal}\n")
    #             f.write(f"Files processed: {len(files)}\n\n")
    #
    #             for file_path in files:
    #                 f.write(f"File: {file_path}\n")
    #                 # Add actual file processing results here
    #                 f.write("  [Processing results would go here]\n\n")
    #
    #         print(f'\033[1;32m[Success]:\033[0m Output written to {option}')
    #     except Exception as e:
    #         print(f'\033[1;31m[Error]:\033[0m Failed to write output file: {str(e)}')
    #
    # # Simulate processing
    # import time
    # for i, file_path in enumerate(files[:5]):  # Process first 5 as example
    #     print(f'\033[1;33m[{i+1}/{min(len(files), 5)}]:\033[0m Processing {file_path.name}...')
    #     time.sleep(0.1)  # Simulate processing time
    #
    # if len(files) > 5:
    #     print(f'\033[1;33m[...]:\033[0m Processed remaining {len(files) - 5} files')
    #
    # print('\033[1;32m[Complete]:\033[0m Inception processing finished!')


def main():
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        display_help()
        sys.exit(1)

    try:
        # Parse arguments using sys.argv
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

        # Find files
        files = find_files(path, extensions)

        if not files:
            print('\033[1;33m[Warning]: No files found matching the criteria\033[0m')

            if extensions:
                print(
                    f'\033[1;33m[Hint]: Try without --ext filter or check if files with extensions {extensions} exist\033[0m')

            sys.exit(0)

        # Process files
        process_files(files, parsed_args['goal'], parsed_args['option'])

    except KeyboardInterrupt:
        print('\n\033[1;31m[Interrupted]: Process cancelled by user\033[0m')
        sys.exit(1)
    except Exception as e:
        print(f'\033[1;31m[Error]: {str(e)}\033[0m')
        sys.exit(1)


if __name__ == "__main__":
    main()