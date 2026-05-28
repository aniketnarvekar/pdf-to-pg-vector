# PDF to PostgreSQL vector store

## Summary

  A command-line interface (CLI) utility built with TypeScript and LangChain.
  This tool extracts text from all PDFs within a specified directory, generates
  vector embeddings using OpenAI, and stores them in a PostgreSQL database (via `pgvector`).

## Prerequisites & Environment Variables

Before running the CLI, you can optionally configure your environment variables.
Creating a `.env` file in the root of your project is the easiest way to manage these.

```bash
# OpenAI Configuration
OPENAI_API_KEY="your_openai_api_key_here" # Required to generate embeddings

# PostgreSQL Configuration
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
POSTGRES_USER="postgres"
POSTGRES_PASS="password"
POSTGRES_DB="vector_db"

# LangChain Configuration
COLLECTION_NAME="pdf_embeddings"
```

## Usage

Run the CLI using:

```bash
pnpm run dev --dirPath=<absolute_path_to_directory>
```

Example:

```bash
pnpm run dev \
  --dirPath=/Users/example/Documents/pdfs \
  --embeddingName=text-embedding-3-small \
  --collectionName=my_pdf_collection
```

---

## CLI Arguments

| Argument | Description |
|---|---|
| `--dirPath` | Absolute path to the directory containing PDF files. |
| `--embeddingName` | OpenAI embedding model name. Default: `text-embedding-3-large`. |
| `--collectionName` | Name of the vector collection. Overrides `COLLECTION_NAME`. Required if `COLLECTION_NAME` is not set. |
| `--postgresHost` | PostgreSQL server host. Overrides `POSTGRES_HOST`. |
| `--postgresPort` | PostgreSQL server port. Overrides `POSTGRES_PORT`. |
| `--postgresUser` | PostgreSQL username. Overrides `POSTGRES_USER`. |
| `--postgresPass` | PostgreSQL password. Overrides `POSTGRES_PASS`. |
| `--postgresDb` | PostgreSQL database name. Overrides `POSTGRES_DB`. |

---

## Notes

- CLI arguments always take precedence over environment variables.
- The application exits with status code `1` if the collection name is not provided through either:
  - `--collectionName`
  - `COLLECTION_NAME`

---

## Tech Stack

- TypeScript
- LangChain
- OpenAI Embeddings
- PostgreSQL
- pgvector
- pnpm
