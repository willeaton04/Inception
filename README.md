# 🎯 Goal of MVP

A single-command CLI tool that:
1. Accepts a folder path.
2. Detects and reads multiple file formats (PDF, DOCX, XLSX, TXT).
3. Uses an LLM to decide which files to process first based on a user-defined goal.
4. Outputs a structured summary to the terminal or file.
5. Stores user context in a vector database

--- 

# 📦 MVP Feature Set

# Inputs
Required: Folder path containing files

## Optional flags:
- --goal "Find Q4 financial data" (scraping objective)
- --ext pdf,docx,xlsx (restrict file types)
- --output summary.json (write to file instead of stdout)
- --local (use local model instead of API)

# Core Functions
1. File scanning
•Recursively scan a directory for supported files.
•Detect file type by extension or magic number.
2. File reading/parsing
•PDFs → pymupdf
•DOCX → python-docx
•XLSX → pandas
•TXT → built-in Python
3. Agentic decision-making
•Feed list of available files + goal into an LLM (e.g., GPT-4.1-mini)
•LLM returns next file to process until all relevant ones are done.
4. Summarization
•Summarize or extract data from each file according to the goal.
•Store result in structured format (JSON/Markdown).
5. CLI UX
•Simple: filescraper ./data --goal "Extract Q4 financial numbers"
•Progress indicator: “Processing file 2 of 7: report_2024.pdf”
•Final output to console or file.

--- 

# Tech Stack
- Language: Python (easy to package with pipx install)
- File Parsing: pymupdf, python-docx, pandas
- LLM API: OpenAI API (agent decision + summarization)
- CLI Framework: click or typer for nice command-line UX
- Packaging: setuptools + pipx or brew formula

--- 

# 🛠 MVP Limitations (Okay for First Release)
- Only supports 4–5 file formats.
- Simple agent loop (pick next file → process → repeat).
