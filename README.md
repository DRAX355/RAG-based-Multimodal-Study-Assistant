# RAG-based Multimodal Study Assistant

An **AI-powered study companion** that processes PDFs, images, and handwritten notes, converts them into a searchable knowledge base, and answers user questions using *Retrieval-Augmented Generation (RAG)* with accurate citations. The system also generates study aids such as flashcards and quizzes.

📌 This project is ideal for students, educators, and lifelong learners who want quick, accurate, and explainable answers based on their own notes.

---

## 🚀 Features

### ✅ Multimodal Input Support
- Upload **PDF documents**, **images**, and **handwritten notes**
- Supports both text-layer PDFs and scanned images

### 📚 Knowledge Extraction
- Uses OCR (Optical Character Recognition) to extract text from images and handwriting
- Preprocesses images for improved accuracy

### 🤖 Smart Retrieval + Generation
- Converts text into **embeddings** (semantic vectors)
- Uses **FAISS** for efficient semantic search
- Reranks results with a cross-encoder for better accuracy
- Answers user questions with an LLM (e.g., OpenAI/Groq) using RAG — grounded in your own notes

### 🎓 Study Aid Generators
- 🔖 **Flashcards** — quick Q/A learning
- ❓ **Quizzes** — simple quiz formulation from notes

### 🧠 Backend + Frontend
- FastAPI backend for ingestion, indexing, and querying
- Streamlit frontend for an interactive UI

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| **OCR / Parsing** | PyMuPDF, pdf2image, Tesseract, TrOCR |
| **Embeddings** | SentenceTransformers |
| **Vector Store** | FAISS (HNSW / Flat) |
| **Reranking** | Cross-Encoder |
| **LLM / Generation** | OpenAI / Groq / Local |
| **Web UI** | Streamlit |
| **Backend API** | FastAPI |

---

## 🧠 How It Works

1. **Extract**  
   User uploads documents/images. The backend extracts text via OCR and direct PDF text extraction.

2. **Transform**  
   Text is cleaned, chunked, and converted into semantic vectors (embeddings). This is your *knowledge base*.

3. **Load**  
   Embeddings and text chunks are stored in a FAISS index for fast semantic retrieval.

4. **Query (RAG)**  
   When a user asks a question:
   - The question is embedded
   - FAISS retrieves relevant chunks
   - They are reranked for relevance
   - The LLM generates an answer using those chunks

5. **Study Aids**  
   From the indexed text, flashcards and quiz questions can be generated.

> This pattern is known as **Retrieval-Augmented Generation (RAG)**, which greatly improves answer accuracy by grounding generative models in user data rather than pre-trained weights alone.

---

## 🛠️ Installation

### 📥 Clone the Repository

```bash
git clone https://github.com/DRAX355/RAG-based-Multimodal-Study-Assistant.git
cd RAG-based-Multimodal-Study-Assistant
```

### 📥 Install Dependencies

Make sure you have Python 3.10+.

```bash
pip install -r requirements.txt
```

### Install External Dependencies

**Poppler** (for PDF images)
- Ubuntu: `sudo apt install poppler-utils`
- macOS: `brew install poppler`
- Windows: Download from [releases](https://github.com/oschwartz10612/poppler-windows/releases)

**Tesseract OCR**
- Ubuntu: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 📡 Environment Setup

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Add your API keys:

```ini
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GROQ_API_URL=https://api.groq.ai/v1
```

---

## 🚀 Running the App

### Start Backend

```bash
uvicorn backend:app --reload --port 8000
```

### Start Frontend

```bash
streamlit run app.py
```

Open your browser and go to:

```
http://localhost:8501
```

---

## 📊 Demo Flow

1. **Upload notes** (PDF / images) in the UI
2. **Build Index** — system extracts text and creates the vector store
3. **Ask a Question** — the assistant retrieves, reranks, and generates answers
4. **See Sources** — retrieved snippets with file + page details
5. **Generate Flashcards / Quiz** — study aids from your notes

### 💡 Example Questions

- "Explain backpropagation based on my notes."
- "Create flashcards from chapter 3."

---

## 📁 Repository Structure

```
├── backend.py          # FastAPI backend server
├── app.py              # Streamlit frontend
├── ocr.py              # OCR processing module
├── parsing.py          # Document parsing utilities
├── embeddings.py       # Embedding generation
├── db.py               # Database/vector store management
├── rag.py              # RAG pipeline implementation
├── rerank.py           # Result reranking
├── agents.py           # AI agent logic
├── tts.py              # Text-to-speech functionality
├── frontend.py         # Frontend components
└── requirements.txt    # Python dependencies
```

---

## 🎯 What Makes This Project Special

✔ Supports handwritten and typed text (multimodal)  
✔ Answers with context-grounded citations (no hallucination)  
✔ Produces study aids automatically  
✔ Designed for students — focused on indexed notes and revision

---

## 📸 Screenshots

### Upload & Indexing
![Upload Screenshot](screenshots/upload.png)

### Ask Question
![Chat Screenshot](screenshots/chat.png)

### Flashcards
![Flashcards Screenshot](screenshots/flashcards.png)

---

## 📌 Future Enhancements

- [ ] Add interactive flashcards with spaced repetition
- [ ] Integrate local LLM support (offline)
- [ ] Add bounding boxes & highlighting in PDFs
- [ ] Extend support to videos & audio

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---





## 🙏 Acknowledgements

Thanks to:

- [OpenAI](https://openai.com/) / [Groq](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- [sentence-transformers](https://www.sbert.net/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 📧 Contact

For questions or feedback, please open an issue or reach out via [GitHub](https://github.com/DRAX355).

---

**⭐ If you find this project helpful, please give it a star!**
