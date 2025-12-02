import pandas as pd
import pickle
from flask import Flask, request, jsonify

# 1. Initialize the App
app = Flask(__name__)

# 2. Load the Model (We do this once when the app starts)
print("Loading model...")
with open(r'random_forest_model/model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully!")

# 3. Define the Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert JSON to Pandas DataFrame (Model expects a DataFrame)
        # We ensure the columns match exactly what the model was trained on
        df = pd.DataFrame(data)
        
        # Make Prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1] # Get probability of Churn
        
        # Return result
        return jsonify({
            'prediction': int(prediction[0]),
            'churn_probability': float(probability[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

# 4. Run the App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)