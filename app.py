from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model
model = None
le = None

def train_model():
    """Train the model on startup"""
    global model, le
    try:
        csv_path = "weather_forecast_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ["Temperature", "Cloud_Cover", "Rain"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        input_x = df[["Temperature", "Cloud_Cover"]].values
        le = LabelEncoder()
        output_y = le.fit_transform(df["Rain"])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(input_x, output_y)
        print("✓ Model trained successfully")
        print(f"  Classes: {le.classes_}")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        raise

# Train model when app starts
train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            cloud = float(request.form['cloud_cover'])
            
            # Validate inputs
            if temp < -50 or temp > 60:
                error = "Temperature should be between -50°C and 60°C"
                return render_template('index.html', prediction=prediction, error=error)
            if cloud < 0 or cloud > 100:
                error = "Cloud Cover should be between 0% and 100%"
                return render_template('index.html', prediction=prediction, error=error)
            
            features = np.array([[temp, cloud]])
            p = model.predict(features)[0]
            prediction = le.inverse_transform([p])[0]
        except ValueError as e:
            error = f"Invalid input: {str(e)}"
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)