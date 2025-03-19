import os
import logging
import fitz  # PyMuPDF for PDF text extraction
import google.generativeai as genai

# Logging setup
logger = logging.getLogger(__name__)

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB9_uEfeyLvJ1O-PrT8Qlj8PlOG-p_MvsU")
genai.configure(api_key=GOOGLE_API_KEY)

# Define generation configuration for Gemini model
generation_config = {
    "temperature": 0.4,
    "max_output_tokens": 200,
}

# Initialize the Gemini model
model_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# Custom TextLoader class for OCR text extraction
class OCRTextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self):
        try:
            text = ""
            with fitz.open(self.file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")

# Function to extract crop details from text using Gemini AI
def extract_crop_details(file_path):
    try:
        # Extract text from PDF
        loader = OCRTextLoader(file_path)
        extracted_text = loader.extract_text()

        # Feed the extracted text to the Gemini model
        prompt = (
            "You are an expert AI extractor tasked with extracting crop details from the given text. "
            "Extract ONLY the following values in this exact order: Nitrogen, Phosphorus, Potassium, temperature, pH, rainfall. "
            "If any value is missing or unclear, use 'N/A' as a placeholder. "
            "Return the values as a single line of comma-separated numbers or placeholders without any additional text or explanation.\n\n"
            f"Text:\n{extracted_text}"
        )

        response = model_gemini.generate_content(prompt)

        # Parse the response
        values = response.text.strip().split(',')
        if len(values) != 6:
            raise ValueError("Invalid response format from Gemini model")

        return {
            'status': 'success',
            'response': {
                'nitrogen': values[0].strip(),
                'phosphorus': values[1].strip(),
                'potassium': values[2].strip(),
                'temperature': values[3].strip(),
                'ph': values[4].strip(),
                'rainfall': values[5].strip()
            }
        }

    except Exception as e:
        logger.error(f"Error extracting crop details: {str(e)}")
        return {'error': str(e)}
