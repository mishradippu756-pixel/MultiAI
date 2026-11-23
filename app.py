import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import database
import ai_service

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_PDF_EXTENSIONS

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
database.init_db()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return 'image'
    elif ext in ALLOWED_PDF_EXTENSIONS:
        return 'pdf'
    return 'unknown'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/status', methods=['GET'])
def get_status():
    api_configured = ai_service.is_api_configured()
    model = ai_service.get_configured_model()
    return jsonify({
        "status": "online",
        "api_configured": api_configured,
        "model": model,
        "name": "MultiAI Multimodal Assistant"
    })

@app.route('/api/settings', methods=['POST'])
def save_settings():
    data = request.get_json() or {}
    api_key = data.get('api_key', '').strip()
    model = data.get('model', '').strip() or 'gemini-2.5-flash'
    
    env_file = os.path.join(BASE_DIR, '.env')
    
    # Read existing env lines
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env_vars[k.strip()] = v.strip()

    if api_key:
        env_vars['GEMINI_API_KEY'] = api_key
        os.environ['GEMINI_API_KEY'] = api_key
    
    env_vars['GEMINI_MODEL'] = model
    os.environ['GEMINI_MODEL'] = model

    with open(env_file, 'w', encoding='utf-8') as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

    return jsonify({
        "success": True,
        "api_configured": ai_service.is_api_configured(),
        "model": model,
        "message": "Settings updated successfully."
    })

# ----------------- CONVERSATION MANAGEMENT ----------------- #

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    conversations = database.get_all_conversations()
    return jsonify({"conversations": conversations})

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    data = request.get_json() or {}
    title = data.get('title', 'New Chat')
    conv = database.create_conversation(title)
    return jsonify(conv), 201

@app.route('/api/conversations/<conv_id>', methods=['GET'])
def get_conversation_details(conv_id):
    conv = database.get_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify(conv)

@app.route('/api/conversations/<conv_id>', methods=['PATCH'])
def rename_conversation(conv_id):
    data = request.get_json() or {}
    new_title = data.get('title', '').strip()
    if not new_title:
        return jsonify({"error": "Title cannot be empty"}), 400
    
    success = database.rename_conversation(conv_id, new_title)
    if not success:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"success": True, "title": new_title})

@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    success = database.delete_conversation(conv_id)
    if not success:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"success": True})

# ----------------- CHAT & MULTIMODAL API ----------------- #

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        conv_id = request.form.get('conversation_id', '').strip()
        message = request.form.get('message', '').strip()
        mode = request.form.get('mode', 'general').strip()
        custom_key = request.form.get('api_key', '').strip()

        # If no conversation exists yet, create one
        if not conv_id:
            conv = database.create_conversation("New Chat")
            conv_id = conv['id']
        else:
            conv = database.get_conversation(conv_id)
            if not conv:
                conv = database.create_conversation("New Chat")
                conv_id = conv['id']

        attachment_info = None
        attachment_path = None
        attachment_type = None
        attachment_name = None

        # Check for uploaded file
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, WEBP, or PDF."}), 400
                
                orig_filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex[:10]}_{orig_filename}"
                saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(saved_path)
                
                file_type = get_file_type(orig_filename)
                attachment_info = {
                    "path": saved_path,
                    "type": file_type,
                    "name": orig_filename,
                    "url": f"/uploads/{unique_name}"
                }
                attachment_path = attachment_info["url"]
                attachment_type = file_type
                attachment_name = orig_filename

        if not message and not attachment_info:
            return jsonify({"error": "Please enter a message or attach a file."}), 400

        # Save user message to database
        user_msg = database.add_message(
            conv_id=conv_id,
            role='user',
            content=message,
            attachment_path=attachment_path,
            attachment_type=attachment_type,
            attachment_name=attachment_name,
            mode=mode
        )

        # Retrieve prior history for context (exclude the current user message just inserted)
        all_msgs = database.get_conversation(conv_id).get('messages', [])
        history = [m for m in all_msgs if m['id'] != user_msg['id']]

        # Generate response using Gemini
        ai_reply_text = ai_service.generate_ai_response(
            conversation_history=history,
            prompt=message,
            attachment_info=attachment_info,
            mode=mode,
            custom_api_key=custom_key
        )

        # Save AI reply to database
        ai_msg = database.add_message(
            conv_id=conv_id,
            role='assistant',
            content=ai_reply_text,
            mode=mode
        )

        # Auto-update conversation title if still default
        title_source = message if message else (f"Analysis of {attachment_name}" if attachment_name else "New Chat")
        database.update_conversation_title_if_default(conv_id, title_source)

        return jsonify({
            "conversation_id": conv_id,
            "user_message": user_msg,
            "ai_message": ai_msg
        })

    except Exception as e:
        app.logger.error(f"Error handling chat: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/api/chat/stream', methods=['POST'])
def handle_chat_stream():
    """
    Server-Sent Events (SSE) streaming chat endpoint for real-time response generation.
    """
    try:
        conv_id = request.form.get('conversation_id', '').strip()
        message = request.form.get('message', '').strip()
        mode = request.form.get('mode', 'general').strip()
        custom_key = request.form.get('api_key', '').strip()

        if not conv_id:
            conv = database.create_conversation("New Chat")
            conv_id = conv['id']
        else:
            conv = database.get_conversation(conv_id)
            if not conv:
                conv = database.create_conversation("New Chat")
                conv_id = conv['id']

        attachment_info = None
        attachment_path = None
        attachment_type = None
        attachment_name = None

        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, WEBP, or PDF."}), 400
                
                orig_filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex[:10]}_{orig_filename}"
                saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(saved_path)
                
                file_type = get_file_type(orig_filename)
                attachment_info = {
                    "path": saved_path,
                    "type": file_type,
                    "name": orig_filename,
                    "url": f"/uploads/{unique_name}"
                }
                attachment_path = attachment_info["url"]
                attachment_type = file_type
                attachment_name = orig_filename

        if not message and not attachment_info:
            return jsonify({"error": "Please enter a message or attach a file."}), 400

        # Save user message
        user_msg = database.add_message(
            conv_id=conv_id,
            role='user',
            content=message,
            attachment_path=attachment_path,
            attachment_type=attachment_type,
            attachment_name=attachment_name,
            mode=mode
        )

        all_msgs = database.get_conversation(conv_id).get('messages', [])
        history = [m for m in all_msgs if m['id'] != user_msg['id']]

        def event_stream():
            full_reply = []
            # Send initial event with metadata
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv_id, 'user_message': user_msg})}\n\n"
            
            try:
                stream_gen = ai_service.generate_ai_response_stream(
                    conversation_history=history,
                    prompt=message,
                    attachment_info=attachment_info,
                    mode=mode,
                    custom_api_key=custom_key
                )
                for chunk in stream_gen:
                    full_reply.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            except Exception as stream_err:
                err_text = f"\n\n*[Stream error: {str(stream_err)}]*"
                full_reply.append(err_text)
                yield f"data: {json.dumps({'type': 'chunk', 'text': err_text})}\n\n"

            final_text = "".join(full_reply).strip() or "No response generated."
            # Save AI message to DB
            ai_msg = database.add_message(
                conv_id=conv_id,
                role='assistant',
                content=final_text,
                mode=mode
            )
            title_source = message if message else (f"Analysis of {attachment_name}" if attachment_name else "New Chat")
            database.update_conversation_title_if_default(conv_id, title_source)
            
            yield f"data: {json.dumps({'type': 'done', 'ai_message': ai_msg, 'conversation_id': conv_id})}\n\n"

        return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

    except Exception as e:
        app.logger.error(f"Error handling stream chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 MultiAI Server starting on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
