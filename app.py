from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load Saved Model
model_data = joblib.load('loan_prediction_model.pkl')

model = model_data['model']
features = model_data['features']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Getting Form Data
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])

        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])

        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])

        credit_history = float(request.form['credit_history'])

        property_area = int(request.form['property_area'])

        # Final Input Data
        input_data = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]])

        # Prediction
        prediction = model.predict(input_data)

        # Result
        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)