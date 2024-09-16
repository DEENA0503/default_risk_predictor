from flask import Flask, request, render_template
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import pandas as pd
import joblib


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

    # convert data to dataFrame with the correct column names
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

    # make prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    print(prediction)
  

    # prepare the result message
    result_message = {
        1: f"The Customer is capable of DEFAULTING. Hence it is RISKY to provide loan! The risk is {prob[1]*100:.2f}%.",
        0: f"The Customer is NOT capable of DEFAULTING. Hence it is POSSIBLE to provide loan! The risk is {prob[1]*100:.2f}%."
    }
    result_short_message = {
        1: "Default Risk !!",
        0: "Safe"
    }

    prediction_result = result_message[prediction]

    # print(type(prediction_result))

    # make graph
    risk=[round(prob[1]*100,2), round(prob[0]*100,2)]
    explode = (0.1, 0)
    l=["Default Risk", "Not Default"]
    colours=["#fc0b03", "#37c414"]

    plt.pie(risk, explode = explode, labels = l, colors=colours,
        autopct = '%1.1f%%',shadow = True, 
        startangle = 90, 
        wedgeprops = {"edgecolor":"black", 
                    'linewidth': 2, 
                    'antialiased': True}) 
    
  
    # saving graph in static file
    plt.savefig(f"static\pie_chart.png", dpi=80)
    plt.clf() 

    return render_template('result.html',prediction=prediction_result, risk=prob[1]*100, id=id, result_short_message=result_short_message[prediction], result=prediction)


if __name__ == "__main__":
    app.run(debug=True)



    
