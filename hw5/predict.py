from flask import Flask, request, jsonify
import pickle
from waitress import serve

app = Flask('credit_score')

# Load your trained model
model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()


    # Prepare the data for prediction
    #features = [job, duration, poutcome]  # features expected by your model

    X = dv.transform([data])

    y_pred = model.predict_proba(X)[0,1].round(3)

    # Make the prediction

    return jsonify({'prediction': y_pred})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port= 8080)

