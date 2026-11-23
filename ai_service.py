import os
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pymupdf  # PyMuPDF

# Load environment variables
load_dotenv()

DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

SYSTEM_PROMPTS = {
    "general": (
        "You are MultiAI, a modern, highly capable multimodal AI assistant with expert intelligence "
        "in analysis, programming, writing, reasoning, and document understanding. "
        "Be clear, accurate, friendly, structured, and helpful. "
        "Use Markdown formatting with clean code blocks, lists, bold highlights, and headers where appropriate."
    ),
    "code": (
        "You are MultiAI's Senior Code & Engineering Assistant. "
        "You are an expert in Python, C, C++, JavaScript, TypeScript, HTML/CSS, SQL, Go, Rust, Java, and systems architecture. "
        "Provide clean, well-commented, idiomatic code, thorough explanations, bug diagnostics, edge case analysis, "
        "and time/space complexity analysis. Always wrap code in proper Markdown code blocks with language identifiers."
    ),
    "writing": (
        "You are MultiAI's Professional Writing Assistant. "
        "Specializing in essays, professional emails, storytelling, creative copywriting, grammar enhancement, "
        "tone adjustment, and executive reports. "
        "Provide compelling, articulate, polished, and context-tailored writing. Adapt your style based on the user's intent."
    ),
    "translate": (
        "You are MultiAI's Universal Multilingual Translator. "
        "Expert in English, Hindi (हिन्दी), Spanish, French, German, Japanese, Chinese, Arabic, and 100+ global languages. "
        "Provide accurate, culturally nuanced, natural-sounding translations. "
        "When appropriate, explain nuances, tone, alternative expressions, or pronunciation guides."
    )
}

def get_api_key(custom_key=None):
    if custom_key and custom_key.strip():
        return custom_key.strip()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None

def get_configured_model():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

def is_api_configured(custom_key=None):
    return bool(get_api_key(custom_key))

def extract_pdf_content(pdf_path, max_chars=100000):
    """
    Extracts text and metadata from a PDF file using PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count
    extracted_pages = []
    total_chars = 0
    truncated = False

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").strip()
        if page_text:
            text_block = f"--- Page {page_num + 1} of {total_pages} ---\n{page_text}"
            if total_chars + len(text_block) > max_chars:
                allowed = max_chars - total_chars
                extracted_pages.append(text_block[:allowed] + "\n...[Document excerpt limit reached]...")
                truncated = True
                break
            else:
                extracted_pages.append(text_block)
                total_chars += len(text_block)
    
    doc.close()
    
    full_text = "\n\n".join(extracted_pages)
    if not full_text.strip():
        full_text = "(The PDF was scanned or does not contain selectable text.)"
        
    return {
        "text": full_text,
        "page_count": total_pages,
        "truncated": truncated
    }

def build_gemini_contents(conversation_history, prompt, attachment_info=None):
    """
    Builds the structured contents list for Gemini API.
    """
    contents = []

    # Add historical messages (limit to last 10 messages for prompt efficiency and context retention)
    recent_history = conversation_history[-10:] if conversation_history else []
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "model"
        text_content = msg.get("content", "").strip()
        if text_content:
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text_content)]
                )
            )

    # Current user turn
    user_parts = []

    # Handle attachments
    if attachment_info:
        file_path = attachment_info.get("path")
        file_type = attachment_info.get("type")
        orig_name = attachment_info.get("name", "Attached File")

        if file_type == "image" and file_path and os.path.exists(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(file_path, "rb") as img_file:
                img_bytes = img_file.read()
            user_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            
            user_text = prompt.strip() if prompt and prompt.strip() else f"Analyze this image ({orig_name}), describe what it shows, and identify key objects and details."
            user_parts.append(types.Part.from_text(text=user_text))

        elif file_type == "pdf" and file_path and os.path.exists(file_path):
            pdf_data = extract_pdf_content(file_path)
            pdf_text = pdf_data["text"]
            pages = pdf_data["page_count"]
            
            instruction = prompt.strip() if prompt and prompt.strip() else "Summarize the key information in this document and provide an executive overview."
            composed_prompt = (
                f"[Attached Document: {orig_name} ({pages} pages)]\n"
                f"```text\n{pdf_text}\n```\n\n"
                f"[User Request]: {instruction}"
            )
            user_parts.append(types.Part.from_text(text=composed_prompt))
    else:
        user_text = prompt.strip() if prompt else "Hello MultiAI!"
        user_parts.append(types.Part.from_text(text=user_text))

    contents.append(
        types.Content(
            role="user",
            parts=user_parts
        )
    )

    return contents

def generate_ai_response(conversation_history, prompt, attachment_info=None, mode="general", custom_api_key=None, model_name=None):
    """
    Generates AI response using Google Gemini API.
    Handles fallback models and error messaging.
    """
    api_key = get_api_key(custom_api_key)
    if not api_key:
        return (
            "⚠️ **Gemini API Key Required**\n\n"
            "To activate MultiAI conversations, please configure your Google Gemini API key:\n\n"
            "1. Click the **Settings ⚙️** icon at the top right of this screen.\n"
            "2. Paste your Google Gemini API Key and click **Save**.\n"
            "3. Or add `GEMINI_API_KEY=your_key_here` to your project's `.env` file.\n\n"
            "*(You can obtain a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey))*."
        )

    client = genai.Client(api_key=api_key)
    contents = build_gemini_contents(conversation_history, prompt, attachment_info)
    system_instruction = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7
    )

    # Determine candidate models
    selected_model = model_name or get_configured_model()
    models_to_try = [selected_model] + [m for m in DEFAULT_MODELS if m != selected_model]

    last_error = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            if response and response.text:
                return response.text
            elif response and response.candidates:
                # Handle possible content in parts
                parts = response.candidates[0].content.parts
                text_pieces = [p.text for p in parts if hasattr(p, "text") and p.text]
                if text_pieces:
                    return "".join(text_pieces)
            return "MultiAI received an empty response. Please try rephrasing your message."
        except APIError as api_err:
            last_error = api_err
            # If model not found or unsupported in region, try next candidate
            err_msg = str(api_err).lower()
            if "not found" in err_msg or "unsupported" in err_msg or "404" in err_msg:
                continue
            else:
                return f"⚠️ **Gemini API Error:** {api_err.message if hasattr(api_err, 'message') else str(api_err)}"
        except Exception as e:
            last_error = e
            # Try next model if applicable
            continue

    return f"⚠️ **Error generating AI response:** {str(last_error)}"

def generate_ai_response_stream(conversation_history, prompt, attachment_info=None, mode="general", custom_api_key=None, model_name=None):
    """
    Streaming generator yielding chunks of text for real-time ChatGPT-like display.
    """
    api_key = get_api_key(custom_api_key)
    if not api_key:
        yield (
            "⚠️ **Gemini API Key Required**\n\n"
            "Please configure your Google Gemini API key via the **Settings ⚙️** icon or in `.env` to start chatting."
        )
        return

    client = genai.Client(api_key=api_key)
    contents = build_gemini_contents(conversation_history, prompt, attachment_info)
    system_instruction = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7
    )

    selected_model = model_name or get_configured_model()
    models_to_try = [selected_model] + [m for m in DEFAULT_MODELS if m != selected_model]

    streamed_anything = False
    for model in models_to_try:
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    streamed_anything = True
                    yield chunk.text
            if streamed_anything:
                return
        except Exception as e:
            if not streamed_anything:
                # Try next model
                continue
            else:
                yield f"\n\n*(Stream interrupted: {str(e)})*"
                return

    if not streamed_anything:
        yield "⚠️ **Error connecting to Gemini API.** Please verify your API key and network connection."
