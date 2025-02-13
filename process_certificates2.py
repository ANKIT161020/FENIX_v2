import requests
import json
import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
import os

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# DeepSeek-R1 API endpoint (local server)
DEEPSEEK_R1_URL = "http://localhost:11434/api/generate"

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_image(image):
    result = reader.readtext(image)
    return " ".join([entry[1] for entry in result])

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def process_with_deepseek(extracted_text):
    """
    Enhanced version to handle DeepSeek's XML-tagged JSON response
    """
    if not extracted_text.strip():
        return {"error": "Empty text extracted from document"}

    prompt = f"""
Extract the following details for the certificate found in the text below:
- Recipient_Name
- Certificate_ID
- Verification_Link (if present)
- Start_Date
- End_Date
- Organization_Name

Return ONLY a valid JSON object between <response> tags with these keys. 
Do NOT include any other text or formatting.

Example valid response:
<response>
{{
  "Recipient_Name": "John Doe",
  "Certificate_ID": "CID-123456",
  "Verification_Link": "",
  "Start_Date": "2023-01-01",
  "End_Date": "2023-12-31",
  "Organization_Name": "ABC Institute"
}}
</response>

Certificate Text:
{extracted_text}
"""

    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(DEEPSEEK_R1_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # Extract JSON from XML-like tags
        response_text = response.json().get("response", "")
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return {"error": "No JSON found in response"}
            
        json_str = response_text[json_start:json_end]
        
        # Parse and validate JSON
        parsed = json.loads(json_str)
        required_fields = [
            "Recipient_Name", "Certificate_ID", "Verification_Link",
            "Start_Date", "End_Date", "Organization_Name"
        ]
        
        return {field: parsed.get(field, "") for field in required_fields}

    except requests.exceptions.RequestException as e:
        return {"error": f"API connection error: {str(e)}"}
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from: {json_str}")
        return {"error": f"Invalid JSON format: {str(e)}"}



def process_certificate(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_extension.lower() in ['.jpg', '.jpeg', '.png']:
        preprocessed_image = preprocess_image(file_path)
        text = extract_text_from_image(preprocessed_image)
    else:
        return {"error": "Unsupported file type"}
    
    text_json = json.dumps({"extracted_text": text})
    
    # Add this before calling DeepSeek
    print(f"OCR Output: {text_json}")

    # Process with DeepSeek
    result = process_with_deepseek(text_json)
    return result if result else {"error": "No data extracted"}

if __name__ == "__main__":
    certificate_folder = ""
    
    for filename in os.listdir(certificate_folder):
        file_path = os.path.join(certificate_folder, filename)
        print(f"Processing: {filename}")
        result = process_certificate(file_path)
        print(json.dumps(result, indent=2))
        print("-" * 50)
