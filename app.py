from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import joblib

# model_path='best_pipeline_xgb.pkl'
# with open(model_path, 'rb') as file:
#     model=pickle.load(file)
# Load your model
model = joblib.load('best_pipeline_lgb.pkl')

app=Flask(__name__)
print(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #extract data from form
    print(request.form.get('numberOfDefaults'))
    data = {
    'First Name': request.form.get('firstName'),
    'Last Name': request.form.get('lastName'),
    'Age': request.form.get('age'),
    'Annual Income': request.form.get('annualIncome'),
    'Home Ownership': request.form.get('homeOwnership'),
    'Employment Length': request.form.get('employmentLength'),
    'Loan Intent': request.form.get('loanIntent'),
    'Loan Grade': request.form.get('loanGrade'),
    'Loan Amount': request.form.get('loanAmount'),
    'Loan Percent Income': request.form.get('loanPercentIncome'),
    'Previous Defaults': request.form.get('previousDefaults'),
    'Credit History': request.form.get('creditHistory')
    }

    # Convert data to DataFrame with the correct column names
    cols = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']
    

    input_data = pd.DataFrame([[
        int(data['Age']), 
        int(data['Annual Income']), 
        data['Home Ownership'],
        float(data['Employment Length']),
        data['Loan Intent'], 
        data['Loan Grade'], 
        int(data['Loan Amount']), 
        round(float(data['Loan Percent Income']) / 100, 2),
        data['Previous Defaults'],
        int(data['Credit History']), 
    ]], columns=cols)

    # Make prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    print(prob)

    # Prepare the result message
    result_message = {
        1: f"The Customer is capable of DEFAULTING. Hence it is RISKY to provide loan! The risk is {prob[1]*100:.2f}%.",
        0: f"The Customer is NOT capable of DEFAULTING. Hence it is POSSIBLE to provide loan! The risk is {prob[1]*100:.2f}%."
    }
    prediction_result = result_message.get(prediction, "Unknown Prediction")
    print(data)

    return render_template('result.html',data=data, prediction=prediction_result)


if __name__ == "__main__":
    app.run(debug=True)



    