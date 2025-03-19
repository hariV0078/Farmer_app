import logging
import tempfile
import os
import imghdr
import cv2
import matplotlib.pyplot as plt
from flask import request, jsonify
from ultralytics import YOLO  
from roboflow import Roboflow
import google.generativeai as genai
 
  # Roboflow SDK


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load YOLOv10 model
def load_yolov10_model(model_path):
    """Load the YOLO model from the given path."""
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to load the YOLO model.") from e

# ✅ Initialize YOLO model
yolov10_model_path = "D:/PROJECTS AND HACKATHONS/Thiran_25/best.pt"
yolov10_model = load_yolov10_model(yolov10_model_path)

# ✅ Initialize Roboflow Client
rf = Roboflow(api_key="QpMf41I94GIAgGcLq33h")
project = rf.workspace().project("plants-diseases-detection-and-classification")
ROBOFLOW_CLIENT = project.version(12).model



def predict_yolo():
    try:
        # ✅ Check if a file is present in the request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # ✅ Check if user wants to use Roboflow
        use_roboflow = request.form.get("use_roboflow", "false").lower() == "true"

        # ✅ Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # ✅ Validate image type using `imghdr`
        file_type = imghdr.what(temp_path)
        if file_type not in ["jpeg", "png"]:
            os.remove(temp_path)
            return jsonify({"error": "Invalid file type. Only JPG and PNG are supported."}), 400

        # ✅ Validate file size (max 5MB)
        MAX_FILE_SIZE_MB = 5
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            os.remove(temp_path)
            return jsonify({"error": "File too large. Max size is 5MB."}), 400

        logger.info(f"Processing file: {temp_path}")

        if use_roboflow:
            # ✅ Use Roboflow Cloud API
            logger.info("Using Roboflow API for inference...")
            roboflow_result = ROBOFLOW_CLIENT.predict(temp_path).json()
            os.remove(temp_path)
            return jsonify({"status": "success", "roboflow_predictions": roboflow_result})
        else:
            # ✅ Use Local YOLOv10 Model
            logger.info("Using local YOLOv10 model for inference...")
            results = yolov10_model(temp_path, conf=0.1)

            # ✅ Convert results to OpenCV image
            annotated_image = results[0].plot()

            # ✅ Display the image with bounding boxes
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

            # ✅ Parse results
            predictions = []
            resultt = []
            for result in results:
                for box in result.boxes:
                    predictions.append({
                        # "class_id": int(box.cls),
                        "class_name": result.names[int(box.cls)],  # Retrieve class label
                        # "confidence": float(box.conf),
                        # "bbox": box.xyxy.tolist(),
                    })
            for result in results:
                for box in result.boxes:
                    resultt.append({
                        "class_name": result.names[int(box.cls)],  # Retrieve class label
                    })

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

            prompt = (
                "You are an expert AI assistant tasked with providing fertilizer and treatment suggestions based on the given text. "
                "The text contains information about a crop disease that has been predicted. "
                "Your task is to analyze the disease and suggest the most suitable fertilizers, nutrients, or treatments to address the issue. "
                "Return ONLY the suggestions as a single line of comma-separated values without any additional text or explanation.\n\n"
                f"Text:\n{resultt}"
            )

            # Generate content using the Gemini model
            response = model_gemini.generate_content(prompt)

            # Extract the text from the GenerateContentResponse object
            response_text = response.text.strip() if hasattr(response, "text") else "N/A"

            return jsonify({"status": "success", "predictions": predictions, "fertilizer_suggestions": response_text})

    except Exception as e:
        logger.error(f"Error during YOLO prediction: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500
    finally:
        # ✅ Ensure the temp file is deleted
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Deleted temporary file: {temp_path}")


