import sqlite3
import uuid
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiai.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        attachment_path TEXT,
        attachment_type TEXT,
        attachment_name TEXT,
        mode TEXT DEFAULT 'general',
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()

def get_all_conversations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, created_at, updated_at,
               (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) as message_count
        FROM conversations
        ORDER BY updated_at DESC
    ''')
    rows = cursor.fetchall()
    conversations = [dict(row) for row in rows]
    conn.close()
    return conversations

def create_conversation(title="New Chat"):
    conv_id = f"chat_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (conv_id, title, now, now))
    conn.commit()
    conn.close()
    return {
        "id": conv_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0
    }

def get_conversation(conv_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM conversations WHERE id = ?', (conv_id,))
    conv_row = cursor.fetchone()
    if not conv_row:
        conn.close()
        return None
    
    cursor.execute('''
        SELECT * FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    ''', (conv_id,))
    messages_rows = cursor.fetchall()
    
    conv_data = dict(conv_row)
    conv_data['messages'] = [dict(m) for m in messages_rows]
    conn.close()
    return conv_data

def rename_conversation(conv_id, new_title):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE conversations
        SET title = ?, updated_at = ?
        WHERE id = ?
    ''', (new_title.strip() or "Untitled Chat", now, conv_id))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def delete_conversation(conv_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def add_message(conv_id, role, content, attachment_path=None, attachment_type=None, attachment_name=None, mode='general'):
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO messages (id, conversation_id, role, content, attachment_path, attachment_type, attachment_name, mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (msg_id, conv_id, role, content, attachment_path, attachment_type, attachment_name, mode, now))
    
    cursor.execute('''
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
    ''', (now, conv_id))
    
    conn.commit()
    conn.close()
    
    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "attachment_path": attachment_path,
        "attachment_type": attachment_type,
        "attachment_name": attachment_name,
        "mode": mode,
        "created_at": now
    }

def update_conversation_title_if_default(conv_id, prompt_text):
    """Auto-generate a neat short title from prompt text if title is 'New Chat'"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
    row = cursor.fetchone()
    if row and row['title'] in ['New Chat', 'Untitled Chat']:
        clean_text = ' '.join(prompt_text.strip().splitlines())
        short_title = clean_text[:40].strip()
        if len(clean_text) > 40:
            short_title += "..."
        if short_title:
            cursor.execute('UPDATE conversations SET title = ? WHERE id = ?', (short_title, conv_id))
            conn.commit()
    conn.close()
