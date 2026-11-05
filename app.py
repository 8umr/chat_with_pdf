"""
Chat With Your PDF
==================
Upload course notes or a handbook and ask questions.
Answers include relevant page references from the uploaded document.
"""

import io
import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from pypdf import PdfReader

# Try importing OpenAI client; if not present, fallback logic handles requests
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished look, citations, and source badges
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .badge-page {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        border: 1px solid #C7D2FE;
        margin-right: 0.35rem;
    }
    .source-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .source-meta {
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .source-snippet {
        color: #475569;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: #FFFFFF;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #CBD5E1;
        white-space: pre-wrap;
    }
    .stat-box {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Document Processing & BM25 Retrieval Engine
# -----------------------------------------------------------------------------
class DocumentChunk:
    """Represents a text chunk tied strictly to a page number."""
    def __init__(self, chunk_id: int, page_num: int, text: str):
        self.chunk_id = chunk_id
        self.page_num = page_num
        self.text = text
        self.tokens = self._tokenize(text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Fast, clean tokenization: lowercase alphanumeric words
        return re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())


class BM25Retriever:
    """
    Lightweight, dependency-free BM25 retrieval engine
    tailored for finding the most relevant document pages/chunks.
    """
    def __init__(self, chunks: List[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_count = len(chunks)
        self.doc_lengths = [len(c.tokens) for c in chunks]
        self.avg_doc_len = (sum(self.doc_lengths) / self.doc_count) if self.doc_count > 0 else 1.0

        # Document frequency per token
        self.df: Dict[str, int] = Counter()
        for c in chunks:
            unique_tokens = set(c.tokens)
            for t in unique_tokens:
                self.df[t] += 1

        # Inverse document frequency
        self.idf: Dict[str, float] = {}
        for t, count in self.df.items():
            self.idf[t] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        q_tokens = DocumentChunk._tokenize(query_text)
        if not q_tokens:
            return [(c, 1.0) for c in self.chunks[:top_k]]

        scores = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            doc_len = self.doc_lengths[i]
            tf = Counter(chunk.tokens)

            for token in q_tokens:
                if token in tf:
                    t_idf = self.idf.get(token, 0.0)
                    t_freq = tf[token]
                    denom = t_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += t_idf * (t_freq * (self.k1 + 1.0)) / denom

            scores.append((chunk, score))

        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


@st.cache_data(show_spinner=False)
def extract_pdf_data(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts pages and chunks from uploaded PDF bytes.
    Cached by file content hash to ensure fast performance.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    pages_data: List[Dict[str, Any]] = []
    chunks: List[DocumentChunk] = []
    total_words = 0
    chunk_counter = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ""
        # Clean extra whitespace
        cleaned_text = re.sub(r"[ \t]+", " ", page_text).strip()
        words = cleaned_text.split()
        total_words += len(words)

        pages_data.append({
            "page_num": page_num,
            "text": cleaned_text,
            "word_count": len(words),
        })

        # Create overlapping chunks if the page is long
        chunk_size = 350
        chunk_overlap = 60
        if len(words) <= chunk_size:
            if cleaned_text:
                chunks.append(DocumentChunk(chunk_counter, page_num, cleaned_text))
                chunk_counter += 1
        else:
            for start in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[start:start + chunk_size]
                chunk_str = " ".join(chunk_words)
                chunks.append(DocumentChunk(chunk_counter, page_num, chunk_str))
                chunk_counter += 1

    return {
        "total_pages": total_pages,
        "total_words": total_words,
        "pages_data": pages_data,
        "chunks": chunks,
    }


# -----------------------------------------------------------------------------
# LLM Integration (xAI / Grok API Client)
# -----------------------------------------------------------------------------
def get_xai_client(api_key: str):
    """
    Initializes standard OpenAI client pointing to the xAI endpoint.
    xAI provides 100% OpenAI-compatible endpoints at https://api.x.ai/v1.
    """
    if not HAS_OPENAI:
        raise ImportError(
            "The 'openai' library is required to connect to the xAI API. "
            "Please run: pip install openai>=1.0.0"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )


def build_system_prompt() -> str:
    return (
        "You are an expert academic and technical assistant designed to answer "
        "questions based strictly on the user's uploaded document (course notes or handbook).\n\n"
        "CORE RULES FOR CITATIONS AND ACCURACY:\n"
        "1. Every factual claim, summary, answer, or insight MUST be attributed to the exact page "
        "number(s) where it appears using the citation format: [Page X] or [Pages X-Y].\n"
        "2. Place citations naturally at the end of relevant sentences or bullet points.\n"
        "3. Only use the provided excerpts from the document. Do not invent facts, page numbers, or assumptions.\n"
        "4. If the provided document excerpts do not contain enough information to answer the question, "
        "explicitly state that the uploaded document does not cover this information.\n"
        "5. Be clear, organized, well-formatted, and helpful. Use bullet points or numbered lists where suitable."
    )


def format_context_prompt(relevant_chunks: List[Tuple[DocumentChunk, float]]) -> str:
    context_sections = []
    seen_pages = set()
    for chunk, _score in relevant_chunks:
        seen_pages.add(chunk.page_num)
        context_sections.append(
            f"--- [DOCUMENT EXCERPT - PAGE {chunk.page_num}] ---\n{chunk.text}\n"
        )
    pages_list = ", ".join(f"Page {p}" for p in sorted(seen_pages))
    header = f"Relevant document excerpts provided below (from {pages_list}):\n\n"
    return header + "\n".join(context_sections)


# -----------------------------------------------------------------------------
# Streamlit Session State & Sidebar
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_doc_hash" not in st.session_state:
    st.session_state.uploaded_doc_hash = None

with st.sidebar:
    st.title("⚙️ Settings & Document")

    # API Key Configuration
    default_key = os.environ.get("XAI_API_KEY", "")
    if hasattr(st, "secrets") and "XAI_API_KEY" in st.secrets:
        default_key = st.secrets["XAI_API_KEY"]

    api_key_input = st.text_input(
        "API Key",
        value=default_key,
        type="password",
        placeholder="xai-...",
        help="Enter your xAI API key. You can get one from the xAI console (https://console.x.ai).",
    )

    if api_key_input:
        st.success("API key loaded", icon="🔑")
    else:
        st.warning("Please provide your API key to chat", icon="⚠️")

    # Model & Generation Parameters
    model_choice = st.selectbox(
        "Model",
        options=[
            "grok-2-latest",
            "grok-2-1212",
            "grok-beta",
            "grok-vision-beta",
        ],
        index=0,
        help="Select the xAI Grok model version to use for answering questions.",
    )

    with st.expander("Advanced Model Settings"):
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Lower values produce more deterministic, factual answers."
        )
        max_tokens = st.slider("Max Response Tokens", min_value=256, max_value=4096, value=2048, step=128)
        top_k_chunks = st.slider(
            "Context Excerpts (Chunks)", min_value=3, max_value=12, value=6,
            help="Number of most relevant page passages to send to the model."
        )

    st.markdown("---")
    st.subheader("📁 Upload Notes / Handbook")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload course notes, syllabus, textbook chapter, or organizational handbook.",
    )

    doc_data = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hash(file_bytes)

        if st.session_state.uploaded_doc_hash != file_hash:
            st.session_state.uploaded_doc_hash = file_hash
            st.session_state.messages = []  # Reset chat on new document upload

        with st.spinner("Extracting and indexing PDF pages..."):
            doc_data = extract_pdf_data(file_bytes)

        # Quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{doc_data['total_pages']}</div>
                    <div class="stat-label">Pages</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{doc_data['total_words']:,}</div>
                    <div class="stat-label">Words</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Page Inspector
        with st.expander("🔍 Page Inspector (Verify text)"):
            page_select = st.number_input(
                "Inspect Page",
                min_value=1,
                max_value=doc_data["total_pages"],
                value=1,
                step=1,
            )
            selected_page_info = doc_data["pages_data"][page_select - 1]
            st.caption(f"Word count on Page {page_select}: {selected_page_info['word_count']}")
            st.text_area(
                f"Page {page_select} Content",
                value=selected_page_info["text"][:1200] + ("..." if len(selected_page_info["text"]) > 1200 else ""),
                height=180,
                disabled=True,
            )

    st.markdown("---")
    # Chat Controls
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        # Export conversation as markdown
        chat_export = "# Chat With Your PDF - Transcript\n\n"
        for m in st.session_state.messages:
            role_name = "User" if m["role"] == "user" else "Assistant"
            chat_export += f"### {role_name}\n\n{m['content']}\n\n"
        st.download_button(
            "💾 Download Chat History",
            data=chat_export,
            file_name="chat_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main Chat Interface
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Chat With Your PDF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Upload course notes or a handbook and ask questions. Answers include relevant page references from the uploaded document.'
    '</div>',
    unsafe_allow_html=True,
)

# Display Hero banner if no file uploaded yet
if uploaded_file is None:
    st.info("👈 Please upload your PDF document in the sidebar to get started.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            ### 1. Upload Document
            Upload lecture notes, course packets, study guides, or company handbooks.
            """
        )
    with col_b:
        st.markdown(
            """
            ### 2. Ask Anything
            Query concepts, policies, exam topics, definitions, or procedural rules.
            """
        )
    with col_c:
        st.markdown(
            """
            ### 3. Page References
            Every answer explicitly cites `[Page X]` so you can cross-check directly.
            """
        )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display cited sources expander if available for assistant message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Cited Sources & Page Excerpts"):
                for s in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-meta">
                                <span class="badge-page">Page {s['page']}</span>
                                <span>Relevance Score: {s['score']:.2f}</span>
                            </div>
                            <div class="source-snippet">{s['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Sample suggested questions if document is ready but chat is empty
if uploaded_file is not None and doc_data and not st.session_state.messages:
    st.markdown("**Suggested questions to get started:**")
    suggested_cols = st.columns(3)
    suggestions = [
        "What are the main topics covered in this document?",
        "Summarize the key definitions or formulas and their pages.",
        "What are the important guidelines, requirements, or deadlines?",
    ]
    for col, suggestion in zip(suggested_cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state.user_suggested_query = suggestion
            st.rerun()

# Handle Suggested Query if clicked
initial_query = None
if "user_suggested_query" in st.session_state:
    initial_query = st.session_state.pop("user_suggested_query")

# Chat input
user_input = st.chat_input("Ask a question about your uploaded document...") or initial_query

if user_input:
    if not api_key_input:
        st.error("Please enter your API key in the sidebar before asking questions.")
    elif uploaded_file is None or doc_data is None:
        st.warning("Please upload a PDF document before asking questions.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant passages using BM25
        with st.spinner("Searching document pages..."):
            retriever = BM25Retriever(doc_data["chunks"])
            top_matches = retriever.query(user_input, top_k=top_k_chunks)

            # Store sources for inspection
            sources_meta = [
                {
                    "page": match.page_num,
                    "score": score,
                    "text": match.text[:350] + ("..." if len(match.text) > 350 else ""),
                }
                for match, score in top_matches
            ]

            context_str = format_context_prompt(top_matches)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                client = get_xai_client(api_key_input)
                system_prompt = build_system_prompt()

                # Build conversation payload
                # Include previous messages for conversational continuity (last 4 turns)
                history_turns = []
                for prev in st.session_state.messages[-5:-1]:
                    history_turns.append({"role": prev["role"], "content": prev["content"]})

                messages_payload = [
                    {"role": "system", "content": system_prompt},
                    *history_turns,
                    {
                        "role": "user",
                        "content": f"{context_str}\n\nUser Question: {user_input}\n\nPlease answer based on the above excerpts and include page citations [Page X].",
                    },
                ]

                # Stream response
                response_stream = client.chat.completions.create(
                    model=model_choice,
                    messages=messages_payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                response_placeholder = st.empty()
                full_response = ""

                for chunk in response_stream:
                    delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
                    if delta:
                        full_response += delta
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Show sources expander
                if sources_meta:
                    with st.expander("📚 View Cited Sources & Page Excerpts"):
                        for s in sources_meta:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-meta">
                                        <span class="badge-page">Page {s['page']}</span>
                                        <span>Relevance Score: {s['score']:.2f}</span>
                                    </div>
                                    <div class="source-snippet">{s['text']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources_meta,
                })

            except Exception as e:
                st.error(f"Error querying model: {str(e)}")