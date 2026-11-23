# 🚀 MultiAI — Multimodal AI Assistant
MultiAI is a full-stack, ChatGPT-inspired Multimodal AI Assistant built with **Python Flask**, **Google Gemini API**, **SQLite**, and modern **vanilla HTML5/CSS3/JavaScript**. It seamlessly handles **real-time text conversations**, **image analysis**, **PDF document extraction & comprehension**, **voice input (speech-to-text)**, and **voice output (text-to-speech)**.
---
## 🌟 Key Features
### 1. 💬 AI Chat
- Conversational chat powered by Google Gemini (`gemini-2.5-flash`, `gemini-1.5-flash`, etc.).
- Maintains full conversation context across turns.
- Clean Markdown formatting, structured lists, tables, blockquotes, and headings.
- Real-time syntax highlighting for code blocks with one-click **Copy Code** buttons.
- Animated thinking/typing indicator.
### 2. 🖼️ Image AI (Vision)
- Supports **JPG, JPEG, PNG, and WEBP** uploads (up to 16MB).
- Ask detailed questions about photos, diagrams, architecture charts, screenshots, and artwork.
- Automatic object identification, scene descriptions, and visual question answering.
- Built-in full-resolution **Image Lightbox** modal.
### 3. 📄 PDF Assistant
- Powered by high-speed **PyMuPDF** (`pymupdf`).
- Extracts text page-by-page across multi-page documents.
- Document summarization, table interpretation, and deep question-answering over PDF contents.
- Document preview card displaying file name and direct link.
### 4. 🎙️ Voice Assistant (Speech-to-Text & Text-to-Speech)
- **Voice Input**: Microphone button utilizing the browser's native **Web Speech API** (`SpeechRecognition` / `webkitSpeechRecognition`).
- Dynamic pulsing audio recording animation and real-time live transcript streaming.
- **Voice Output**: Seamless speech synthesis using the browser's **Speech Synthesis API** (`speechSynthesis`).
- Per-message **Read Aloud** button and global **Stop Speaking** floating banner with animated sound wave bars.
- Customizable speech rate (0.5x – 2.0x) and voice selector in Settings.