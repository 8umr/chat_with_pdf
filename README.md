# Chat With Your PDF

> **Upload course notes or a handbook and ask questions. Answers include relevant page references from the uploaded document.**

A streamlined, high-performance Streamlit web application that lets students, researchers, and professionals upload PDF documents (lecture notes, academic papers, course syllabi, training manuals, or organizational handbooks) and chat naturally with them.

The application integrates with the official **xAI API** (`grok-2-latest`, `grok-beta`) with a built-in page-level BM25 retrieval engine. Every answer provides verifiable citations formatted as `[Page X]` or `[Pages X-Y]`, paired with an interactive source viewer to inspect exact excerpts and verify accuracy.

---

## Key Features

- 📄 **Page-Aware PDF Ingestion**: Extracts text page-by-page with exact page numbering and word counts using `pypdf`.
- 🔍 **BM25 Semantic Retrieval**: Automatically identifies and ranks the most relevant sections across large documents without requiring external vector databases or complex setup.
- 📌 **Exact Page References**: System prompt enforces strict provenance—every claim and insight is tied back to its source page (`[Page X]`).
- ⚡ **xAI API Integration**: Connects directly to xAI's OpenAI-compatible endpoint (`https://api.x.ai/v1`) using `grok-2-latest` or `grok-beta` for fast, intelligent reasoning with large context windows.
- 🌊 **Real-Time Streaming**: Responses stream live into the chat interface for a responsive conversational experience.
- 🔎 **Built-in Page Inspector**: Browse and inspect full text from any page of the uploaded PDF directly in the sidebar.
- 💾 **Transcript Export**: Download your full Q&A session as a clean Markdown transcript.
- 🔒 **Secure Key Handling**: Enter your API key securely in the UI or load it via `.env` or `.streamlit/secrets.toml`.

---

## Project Structure

```text
chat_with_pdf/
├── app.py              # Main Streamlit web application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template for API key
└── README.md           # Documentation and setup instructions
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- An xAI API key (starts with `xai-...`). You can obtain one from the [xAI Console](https://console.x.ai).

### 2. Create and Activate a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
You can provide your API key in any of the following three ways:

#### Option A: In the Web Application (Easiest)
Simply start the app and paste your API key into the password field in the sidebar.

#### Option B: Using a `.env` File
Create a `.env` file in the project folder (or copy `.env.example`):
```bash
cp .env.example .env
```
Edit `.env` and paste your key:
```ini
XAI_API_KEY=xai-your-actual-api-key-here
```

#### Option C: Streamlit Secrets (for Streamlit Community Cloud)
Create `.streamlit/secrets.toml`:
```toml
XAI_API_KEY = "xai-your-actual-api-key-here"
```

---

## Running the Application

Launch the Streamlit app with:
```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`.

---

## How It Works

1. **Document Parsing**: When you upload a PDF, the app extracts text page by page. Long pages are divided into overlapping chunks while strictly preserving their source page numbers.
2. **Relevance Retrieval (BM25)**: When you ask a question, a local Okapi BM25 engine scores all document chunks against your question terms and selects the top most relevant excerpts.
3. **Grounded Prompting**: The app constructs a prompt containing the retrieved page passages and instructs the model to answer strictly based on those excerpts while citing `[Page X]` for every statement.
4. **Streaming & Verification**: The response streams in real time. Underneath each response, an expandable **"📚 View Cited Sources & Page Excerpts"** drawer shows the exact text passages and relevance scores retrieved for that turn.

---

## Available Models

In the sidebar, you can select from the following xAI models:
- `grok-2-latest` (Recommended: High intelligence, fast, and balanced)
- `grok-2-1212` (Specific stable snapshot)
- `grok-beta` (General xAI reasoning model)
- `grok-vision-beta` (Multimodal variant)

---

## Troubleshooting

- **Authentication Error**: Double check that your API key is correct and active in the xAI console.
- **PDF text not extracting**: Some PDFs are scanned images without an OCR text layer. If pages show 0 words, run OCR on the PDF beforehand.
- **Streamlit port conflict**: If port 8501 is occupied, run `streamlit run app.py --server.port 8502`.