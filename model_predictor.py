import numpy as np
import pickle

class ModelPredictor:
    def __init__(self, model_path='crop_planner_model.pkl'):
        self.model = self.load_model(model_path)
    
    # Function to load the model from a .pkl file
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    # Function to predict the output based on input data
    def predict(self, input_data):
        # Convert the input data into a numpy array and reshape it for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Predict the output using the loaded model
        prediction = self.model.predict(input_array)
        
        # Debugging: Print the type and shape of the prediction
        print(f"Prediction type: {type(prediction)}, Prediction shape: {getattr(prediction, 'shape', 'N/A')}")

        # Ensure the prediction is a 1D array or list
        if isinstance(prediction, np.ndarray):
            prediction = prediction.flatten()  # Flatten to 1D array
            prediction = prediction.tolist()   # Convert to Python list
        
        # Map the prediction to human-readable values
        suggested_crop = prediction[0]
        sowing_month = prediction[1]
        harvesting_month = prediction[2]
        irrigation = prediction[3]
        
        # Return the mapped values as a dictionary
        return {
            "suggested_crop": suggested_crop,
            "sowing_month": sowing_month,
            "harvesting_month": harvesting_month,
            "irrigation": irrigation
        }