# Day 1 – Project Setup

Create Git repo & folder structure (cli/, scanner/, reader/, agent/, summarizer/, output/, db/).

Initialize pyproject.toml or setup.py for packaging.

Install dependencies: typer, pymupdf, python-docx, pandas, openai, chromadb (or faiss), tqdm.

Set up .env for API keys.

# Day 2 – CLI Entry Point

Implement CLI with typer:

bash
Copy
Edit
filescraper ./data --goal "Find Q4 financial data" --ext pdf,docx --output summary.json
Parse args: folder path, goal, extensions, output, local mode.

# Day 3 – File Scanner

Implement recursive file scanning with filtering by extensions.

Add support for magic number detection (optional for MVP).

Unit test scanning on a test folder.

# Day 4 – TXT & DOCX Reader

Implement read_txt(path) and read_docx(path) functions returning plain text.

Add simple tests for both.

# Day 5 – PDF Reader

Implement read_pdf(path) using pymupdf.

Handle multi-page extraction.

Test with sample PDFs.

# Day 6 – XLSX Reader

Implement read_xlsx(path) using pandas.read_excel.

Convert to CSV string or structured table text for LLM.

Test with a real spreadsheet.

# Day 7 – Reader Integration

Create a read_file(path) wrapper that:

Detects extension

Calls correct reader

Returns text

Unit test with all formats.

Week 2 – Agent & Summarization
Goal: Implement LLM decision loop & summarizer.

# Day 8 – Agent Decision Function

Implement choose_next_file(files, goal):

Sends list of file names + goal to OpenAI API

Returns filename to process.

Add dummy LLM mock for testing without API cost.

# Day 9 – Summarization Function

Implement summarize_file(text, goal):

Sends file text + goal to LLM.

Returns structured JSON.

Keep format consistent (e.g., { "file": "name", "summary": "...", "data": {...} }).

# Day 10 – Looping Through Files

Implement loop:

Agent picks next file.

Read file.

Summarize.

Store result.

Remove from list.

Add progress indicator with tqdm.

# Day 11 – Vector Database Setup

Use chromadb for simplicity.

Store:

File name

Goal

Summary text

Embedding

Implement add_to_db(file_summary).

# Day 12 – Local Model Mode

Add --local flag to switch to Ollama or Hugging Face pipeline for decision + summarization.

Keep API calls abstracted.

# Day 13 – Output Handler

Implement:

JSON output

Stdout printing

Respect --output flag.

# Day 14 – First Full Run

Run full pipeline on a test folder.

Debug API calls, parsing errors, encoding issues.

Week 3 – Polishing & Testing
Goal: Make it reliable, documented, and packaged.

# Day 15 – Error Handling

Handle:

Corrupted files

Empty files

API errors (retry with exponential backoff)

Skip problematic files gracefully.

# Day 16 – Logging

Add --verbose flag.

Log steps: scanning, reading, LLM decisions, output writing.

# Day 17 – CLI UX Improvements

Improve progress bar:

yaml
Copy
Edit
Processing 2/7: report_Q4.pdf
Colorize output with rich.

# Day 18 – Packaging

Make package installable via pipx install ..

Test on fresh environment.

# Day 19 – Documentation

Create README.md:

Install instructions

Example commands

Supported formats

Local vs API usage

# Day 20 – Extended Testing

Test on large folders (100+ files).

Test all flags combinations.

Test local vs API.

# Day 21 – Final Polish & Release

Tag release in Git.

Optionally publish to PyPI.

Share internally for feedback.