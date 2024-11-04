from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    gb = pickle.load(f)

crop_mapping = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Prepare input data for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make a prediction using the model
        prediction = gb.predict(input_data)
        
        # Map the prediction to the crop name
        crop = crop_mapping.get(prediction[0], 'Unknown')
        
        # Render the result page with the prediction
        return render_template('result.html', prediction=crop)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
