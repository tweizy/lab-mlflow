import joblib
import json
import os
import pandas as pd
import logging

# Called when the service is initialized
def init():
    global model, scaler
    
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    
    model_path = os.path.join(model_dir, "model_files/model.pkl")
    scaler_path = os.path.join(model_dir, "model_files/scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    logging.info("Model and scaler loaded successfully.")

def run(raw_data):
    try:
        data_dict = json.loads(raw_data)
        
        data = pd.DataFrame(
            data_dict["dataframe_split"]["data"],
            columns=data_dict["dataframe_split"]["columns"]
        )
        
        scaled_data = scaler.transform(data)
        
        prediction = model.predict(scaled_data)
        
        return json.dumps({"prediction": prediction.tolist()})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return json.dumps({"error": str(e)}), 400