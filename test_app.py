import unittest
import os
import json
import io
import pymupdf
from PIL import Image

# Ensure path
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import database
import ai_service
from app import app

class MultiAITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        database.init_db()

    def test_01_status_endpoint(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'online')
        self.assertIn('model', data)

    def test_02_conversation_crud(self):
        # Create
        create_res = self.client.post('/api/conversations', json={"title": "Test Chat Alpha"})
        self.assertEqual(create_res.status_code, 201)
        conv = create_res.get_json()
        conv_id = conv['id']
        self.assertEqual(conv['title'], "Test Chat Alpha")

        # List
        list_res = self.client.get('/api/conversations')
        self.assertEqual(list_res.status_code, 200)
        convs = list_res.get_json().get('conversations', [])
        self.assertTrue(any(c['id'] == conv_id for c in convs))

        # Get
        get_res = self.client.get(f'/api/conversations/{conv_id}')
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.get_json()['id'], conv_id)

        # Rename
        rename_res = self.client.patch(f'/api/conversations/{conv_id}', json={"title": "Renamed Chat Alpha"})
        self.assertEqual(rename_res.status_code, 200)
        self.assertEqual(rename_res.get_json()['title'], "Renamed Chat Alpha")

        # Delete
        delete_res = self.client.delete(f'/api/conversations/{conv_id}')
        self.assertEqual(delete_res.status_code, 200)

        # Verify deletion
        get_after = self.client.get(f'/api/conversations/{conv_id}')
        self.assertEqual(get_after.status_code, 404)

    def test_03_pdf_processing_with_pymupdf(self):
        # Create a sample PDF in memory/file
        test_pdf_path = os.path.join(BASE_DIR, 'test_sample.pdf')
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 72), "MultiAI PyMuPDF Integration Test Document\nArtificial Intelligence & Data Processing")
        doc.save(test_pdf_path)
        doc.close()

        # Extract using ai_service
        extracted = ai_service.extract_pdf_content(test_pdf_path)
        self.assertEqual(extracted['page_count'], 1)
        self.assertIn("MultiAI PyMuPDF Integration Test", extracted['text'])

        # Clean up
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)

    def test_04_chat_without_api_key_returns_guidance(self):
        # Chat without setting API key should return friendly guidance
        response = self.client.post('/api/chat', data={
            'message': 'Hello MultiAI',
            'mode': 'general'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('ai_message', data)
        self.assertIn('Gemini API Key Required', data['ai_message']['content'])

    def test_05_chat_with_image_upload(self):
        # Generate an in-memory test image
        img_byte_arr = io.BytesIO()
        img = Image.new('RGB', (100, 100), color='indigo')
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        response = self.client.post('/api/chat', data={
            'message': 'What color is this image?',
            'mode': 'general',
            'file': (img_byte_arr, 'test_image.png')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('user_message', data)
        self.assertEqual(data['user_message']['attachment_type'], 'image')
        self.assertTrue(data['user_message']['attachment_path'].startswith('/uploads/'))

    def test_06_chat_with_pdf_upload(self):
        # Generate an in-memory PDF
        test_pdf_path = os.path.join(BASE_DIR, 'test_upload.pdf')
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "MultiAI System Architecture Report 2026")
        doc.save(test_pdf_path)
        doc.close()

        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = io.BytesIO(f.read())

        response = self.client.post('/api/chat', data={
            'message': 'Summarize this document',
            'mode': 'general',
            'file': (pdf_bytes, 'system_report.pdf')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('user_message', data)
        self.assertEqual(data['user_message']['attachment_type'], 'pdf')
        self.assertTrue(data['user_message']['attachment_path'].startswith('/uploads/'))

        # Clean up
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)

if __name__ == '__main__':
    unittest.main()
