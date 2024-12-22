import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load the model once
model = pickle.load(open("model.sav", "rb"))

# Load the dataset once (this should contain all the features used to train the model)
df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Retrieve form inputs and convert them to appropriate types
        inputQuery1 = int(request.form['query1'])  # SeniorCitizen as int
        inputQuery2 = float(request.form['query2'])  # MonthlyCharges as float
        inputQuery3 = float(request.form['query3'])  # TotalCharges as float
        inputQuery4 = request.form['query4']  # gender
        inputQuery5 = request.form['query5']  # Partner
        inputQuery6 = request.form['query6']  # Dependents
        inputQuery7 = request.form['query7']  # PhoneService
        inputQuery8 = request.form['query8']  # MultipleLines
        inputQuery9 = request.form['query9']  # InternetService
        inputQuery10 = request.form['query10']  # OnlineSecurity
        inputQuery11 = request.form['query11']  # OnlineBackup
        inputQuery12 = request.form['query12']  # DeviceProtection
        inputQuery13 = request.form['query13']  # TechSupport
        inputQuery14 = request.form['query14']  # StreamingTV
        inputQuery15 = request.form['query15']  # StreamingMovies
        inputQuery16 = request.form['query16']  # Contract
        inputQuery17 = request.form['query17']  # PaperlessBilling
        inputQuery18 = request.form['query18']  # PaymentMethod
        inputQuery19 = int(request.form['query19'])  # tenure as int

        # Prepare the input data into a DataFrame
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
                 inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
                 inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

        new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                             'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                             'PaymentMethod', 'tenure'])

        # Add the tenure_group based on the tenure value
        max_tenure_value = 72  # Adjust this if necessary
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, max_tenure_value, 12)]
        new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, max_tenure_value + 10, 12), right=False, labels=labels)

        # Drop the original 'tenure' column
        new_df.drop(columns=['tenure'], inplace=True)

        # Create dummy variables for categorical columns
        categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                               'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
        new_df_dummies = pd.get_dummies(new_df[categorical_columns])

        # Ensure the model's columns are used in the final input DataFrame
        model_columns = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
            'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
            'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No',
            'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
            'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month',
            'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'tenure_group_1 - 12',
            'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48', 'tenure_group_49 - 60',
            'tenure_group_61 - 72'
        ]

        # Ensure the DataFrame has all required columns, add missing ones with 0
        for column in model_columns:
            if column not in new_df_dummies.columns:
                new_df_dummies[column] = 0

        # Reorder columns to match the model's expected order
        new_df_dummies = new_df_dummies[model_columns]

        # Make the prediction
        prediction = model.predict(new_df_dummies)
        probablity = model.predict_proba(new_df_dummies)[:, 1]

        if prediction == 1:
            o1 = "This customer is likely to be churned!!"
            o2 = "Confidence: {}".format(probablity * 100)
        else:
            o1 = "This customer is likely to continue!!"
            o2 = "Confidence: {}".format(probablity * 100)

        return render_template('home.html',
                               output1=o1,
                               output2=o2,
                               query1=request.form['query1'],
                               query2=request.form['query2'],
                               query3=request.form['query3'],
                               query4=request.form['query4'],
                               query5=request.form['query5'],
                               query6=request.form['query6'],
                               query7=request.form['query7'],
                               query8=request.form['query8'],
                               query9=request.form['query9'],
                               query10=request.form['query10'],
                               query11=request.form['query11'],
                               query12=request.form['query12'],
                               query13=request.form['query13'],
                               query14=request.form['query14'],
                               query15=request.form['query15'],
                               query16=request.form['query16'],
                               query17=request.form['query17'],
                               query18=request.form['query18'],
                               query19=request.form['query19'])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
