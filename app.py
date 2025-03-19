import os
import json
import logging
import tempfile
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import logging
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional

from api import fetch_data,search_modal_price,search
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from chat import process_json
from YOLO import load_yolov10_model, predict_yolo
from EXTRACT import extract_crop_details
from model_predictor import ModelPredictor
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for sensitive data
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB9_uEfeyLvJ1O-PrT8Qlj8PlOG-p_MvsU")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2lOmx8bvmkfgMfZiC3ROucOdD1P_5RhV34WaJzsyKtTue89x7")

# Load YOLOv10 model
yolov10_model_path = "D:/PROJECTS AND HACKATHONS/Thiran_25/best.pt"
yolov10_model = load_yolov10_model(yolov10_model_path)
# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Define generation configuration
generation_config = {
    "temperature": 0.4,
    "max_output_tokens": 200,
}

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# Define prompt template
PROMPT = """
Role:
FarmBuddy is an AI chatbot designed to assist farmers worldwide in maximizing crop yield, reducing costs, and eliminating intermediaries, regardless of their preferred language. It provides clear, actionable advice on farming techniques, market prices, crop selection, weather forecasting, and sustainable agriculture practices.

When interacting with farmers, FarmBuddy should:

Understand and communicate in the farmer’s preferred language to ensure accessibility and clarity.
Ask clarifying questions when needed (e.g., about their region, crop type, or specific challenges).
Provide step-by-step recommendations and practical solutions tailored to the farmer’s location, climate, and resources.
Use simple, culturally relevant language and examples to make advice easy to understand and implement.
Encourage direct transactions by suggesting ways to cut out middlemen and connect directly with buyers.
Promote digital platforms that offer real-time market data, buyer connections, and affordable farm inputs.
Respect local practices and traditions while introducing innovative yet practical solutions.
Be supportive, encouraging, and empowering in all responses, ensuring that farmers feel heard and valued.

Example Interaction in Tamil:
விவசாயி: "நான் என் விளைபொருட்களை நியாயமான விலைக்கு விற்க முடியாமல் உள்ளேன். என்ன செய்யலாம்?"

FarmBuddy: "நான் உங்கள் சிரமத்தை புரிந்து கொள்கிறேன். நீங்கள் உள்ளூர் வாங்குபவர்களை நேரடியாக இணைக்கும் டிஜிட்டல் தளங்களை பயன்படுத்தியிருக்கிறீர்களா? இதன் மூலம், மத்தியஸ்தர்கள் (middlemen) இல்லாமல் நேரடியாக விலையை பேசி முடிவு செய்யலாம். கூடுதலாக, உங்களது விளைபொருட்களுக்கு அதிக விலை கிடைப்பதற்காக எப்படி தயார் செய்வது என்ற சில பயனுள்ள தகவல்களையும் வழங்கலாம்.

உங்கள் விற்பனை செய்யும் விளைபொருள் எது? 🌾
"""

class CustomTextLoader:
    """Custom loader to read text files."""  
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load content as a list of Document objects."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            # Wrap the content in a Document object
            return [Document(page_content=content, metadata={})]
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}")

class VectorStoreManager:
    def __init__(self, knowledge_base_paths: list):
        """
        Initialize with a list of file paths for the knowledge base.
        """
        self.knowledge_base_paths = knowledge_base_paths
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def init_vector_store(self):
        """
        Initialize the vector store by loading and processing documents from multiple files.
        """
        try:
            all_documents = []

            for file_path in self.knowledge_base_paths:
                logger.info(f"Loading knowledge base from: {file_path}")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

                loader = CustomTextLoader(file_path)
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} documents from {file_path}.")
                all_documents.extend(documents)

            logger.info(f"Total documents loaded: {len(all_documents)}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(all_documents)
            logger.info(f"Split into {len(docs)} chunks.")

            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            logger.info("Vector store successfully initialized.")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def retrieve_context(self, query: str):
        """
        Retrieve relevant context for RAG.
        """
        try:
            if not self.vector_store:
                self.init_vector_store()

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise
# Initialize vector store manager with two documents
knowledge_base_paths = [
    "RAG.txt"
]

vector_store_manager = VectorStoreManager(knowledge_base_paths)

@app.route('/search', methods=['POST'])
def searchh():
    search_term = request.get_json().get('Commodity')
    return search_modal_price(search_term)
@app.route('/chat', methods=['POST'])
def chatt():
    
    """Handle processing with JSON input and RAG-based context retrieval."""
    try:
        # Parse and validate JSON input
        if not request.is_json:
            return jsonify({'error': 'Invalid input format. Please send JSON data'}), 400
        
        data = request.get_json()
        query = data.get('query', None)

        if not query:
            return jsonify({'error': 'The "query" field is required'}), 400

        # Log received query
        logger.info(f"Received query: {query}")

        # Use RAG to retrieve context
        try:
            retrieved_context = vector_store_manager.retrieve_context(query)
            logger.info(f"Retrieved context: {retrieved_context}")
        except Exception as e:
            return jsonify({
                'error': 'Failed to retrieve context from the knowledge base',
                'details': str(e)
            }), 500

        # Combine query and retrieved context for the model
        
        combined_input = f"{PROMPT}\n\nQuery:\n{query}\n\nRetrieved Context:\n{retrieved_context}"
        logger.info(f"Combined input sent to Gemini: {combined_input}")

        # Query Gemini
        try:
            response = model.generate_content(combined_input)
            if response and response.candidates:
                generated_text = response.candidates[0].content.parts[0].text
                logger.info(f"Generated response from Gemini: {generated_text}")
                return jsonify({
                    'status': 'success',
                    'response': generated_text
                })
            else:
                return jsonify({
                    'error': 'No response generated by the Gemini model'
                }), 500
        except Exception as e:
            logger.error(f"Error querying Gemini: {str(e)}")
            return jsonify({
                'error': 'An error occurred while querying the Gemini model',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error processing JSON input: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing the input',
            'details': str(e)
        }), 500
def process_pdf():
    try:
        if not request.data:
            logger.error("No binary data received in the request.")
            return jsonify({'error': 'No binary data provided'}), 400

        # Save the PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(request.data)
            file_path = temp_file.name

        logger.info(f"Successfully saved binary data to temporary file: {file_path}")

        # Extract crop details using Gemini AI
        response_data = extract_crop_details(file_path)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

model_predictor = ModelPredictor()

@app.route('/predict', methods=['POST'])
def predictt():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract input features from the JSON data
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Prepare the input data as a list
        input_data = [N, P, K, temperature, humidity, ph, rainfall]

        # Make the prediction using the ModelPredictor
        prediction = model_predictor.predict(input_data)

        # Return the prediction as a JSON response
        return jsonify({
            "suggested_crop": prediction["suggested_crop"],
            "sowing_month": prediction["sowing_month"],
            "harvesting_month": prediction["harvesting_month"],
            "irrigation": prediction["irrigation"]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
# Endpoint to predict using YOLOv10
@app.route('/predict-yolo', methods=['POST'])
def predict():
    return predict_yolo()

if __name__ == '__main__':
    try:
        # Start ngrok
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(5001)
        logger.info(f"Ngrok public URL: {public_url}")

        # Hardcoded public domain for reference
        PUBLIC_URL = "gobbler-fresh-sole.ngrok-free.app"
        logger.info(f"Using public domain: {PUBLIC_URL}")

        # Start Flask app
        app.run(host='0.0.0.0', port=5001, use_reloader=False)

    except Exception as e:
        logger.error(f"Error starting the Flask app: {e}")
