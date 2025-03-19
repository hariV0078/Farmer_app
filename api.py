from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import pandas as pd
import os
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# Path to the CSV file
file_path = "D:/PROJECTS AND HACKATHONS/Thiran_25/9ef84268-d588-465a-a308-a864a43d0070.csv"

@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    """Confirm the existence of the CSV file in the local directory."""
    if os.path.exists(file_path):
        return jsonify({"message": "CSV file exists."}), 200
    else:
        return jsonify({"error": "CSV file not found."}), 404

def search_modal_price(search_term):
    """Search for the modal price of a specific commodity in the locally saved CSV file."""
    if not os.path.exists(file_path):
        abort(404, description="CSV file not found. Please fetch the data first.")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        abort(500, description=f"Error reading CSV file: {str(e)}")

    result = df[df['Commodity'].str.contains(search_term, case=False, na=False)]
    
    if not result.empty:
        modal_price = int(result.iloc[0]['Modal_x0020_Price'])
        return {"Commodity": search_term, "Modal Price": modal_price}
    else:
        return None

@app.route('/search', methods=['POST'])
def search():
    search_term = request.get_json().get('Commodity')
    
    if not search_term:
        return jsonify({"error": "No search term provided."}), 400
    
    result = search_modal_price(search_term)
    if result:
        return jsonify(result), 200
    else:
        return jsonify({"message": f"No results found for '{search_term}'."}), 404

if __name__ == "__main__":
    # Start ngrok and expose the Flask app
    public_url = ngrok.connect(5000)
    print(f"Flask app is running at: {public_url}")
    
    # Run the Flask app
    app.run(port=5000, debug=True)
