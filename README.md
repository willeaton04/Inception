# Agentic Question-Answering System

This project is a sophisticated, agentic question-answering system that leverages AI to analyze a codebase or a directory of files and provide intelligent answers to user queries. It uses Ollama's language models to understand questions, analyze file content, and synthesize comprehensive answers. A vector database, powered by ChromaDB, is used for efficient semantic search to find the most relevant information within the files.

## Features

*   **Natural Language Questions:** Ask questions about your codebase in plain English.
*   **Intelligent File Analysis:** The system intelligently identifies relevant files and analyzes their content to answer your questions.
*   **Semantic Search:** Utilizes a vector database (ChromaDB) to perform semantic searches, finding relevant code snippets and text even if they don't contain the exact keywords from your question.
*   **AI-Powered Synthesis:** Leverages Ollama's powerful language models (e.g., gemma2:2b) to synthesize information from multiple sources and provide a coherent, easy-to-understand answer.
*   **Command-Line Interface:** A user-friendly CLI for interacting with the system.
*   **Extensible:** The system is designed to be extensible, allowing for the addition of new functionalities and integrations.

## How It Works

The system follows these steps to answer a question:

1.  **Question Analysis:** When you ask a question, the system first analyzes it to understand your intent, the key concepts you're interested in, and the relevant file types.
2.  **File Prioritization:** Based on the question analysis, the system prioritizes the files in the specified directory, ranking them by their likely relevance to your question.
3.  **File Analysis and Embedding:** The system then reads the content of the most relevant files. For each file, it:
    *   Chunks the content into smaller, manageable pieces.
    *   Generates embeddings (vector representations) for each chunk using a sentence-transformer model.
    *   Stores these embeddings in a ChromaDB vector database.
4.  **Semantic Search:** The system performs a semantic search on the vector database to find the chunks of text that are most semantically similar to your question.
5.  **Answer Synthesis:** The system sends the most relevant file insights and semantic search results to an Ollama language model. The model then synthesizes this information to generate a comprehensive answer to your question.
6.  **Result Presentation:** The final answer, along with supporting information such as the files that were used to generate the answer, is presented to you in the console.

## Usage

To use the system, run the `main.py` script from your terminal.

### Prerequisites

*   Python 3.7+
*   Ollama installed and running. You can download it from [https://ollama.ai/](https://ollama.ai/).

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

```bash
python main.py <filepath> --question "Your question about the files"
```

**Arguments:**

*   `<filepath>`: The path to the directory or file you want to analyze.
*   `--question` or `-q`: The question you want to ask.

**Optional Arguments:**

*   `--ext` or `-e`: A comma-separated list of file extensions to include in the analysis (e.g., "py,js,md").
*   `--output` or `-o`: The path to a file where you want to save the results in JSON format.
*   `--model` or `-m`: The Ollama model to use (e.g., "gemma2:2b").

### Example

```bash
python main.py . --question "What is the main purpose of this project?"
```

## Configuration

The system can be configured through command-line arguments. For more details on the available options, run:

```bash
python main.py --help
```

## Dependencies

The project relies on the following Python packages:

*   `ollama`: For interacting with the Ollama API.
*   `chromadb`: For the vector database.
*   `sentence-transformers`: For generating text embeddings.
*   `numpy`: For numerical operations.

You can install all the dependencies by running `pip install -r requirements.txt`.