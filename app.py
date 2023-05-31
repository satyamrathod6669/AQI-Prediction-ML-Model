from flask import Flask, render_template,url_for,request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

app = Flask(__name__)
@app.route('/user')
def user():
    return render_template("userInput.html")

@app.route('/answer',methods=['POST','GET'])
def answer():
    if request.method == 'POST':
        pm25 = request.form.get("pm25")
        pm10 = request.form.get("pm10")
        no = request.form.get("no")
        no2 = request.form.get("no2")
        nox = request.form.get("nox")
        nh3 = request.form.get("nh3")
        co = request.form.get("co")
        so2 = request.form.get("so2")
        o3 = request.form.get("o3")
        benzene = request.form.get("benzene")
        toluene = request.form.get("toluene")
        xylene = request.form.get("xylene")
        datainfo={"PM2.5":pm25,"PM10":pm10,"NO":no,"NO2":no2,"NOx":nox,"NH3":nh3,"CO":co,"SO2":so2,"O3":o3,"Benzene":benzene,"Toluene":toluene,"Xylene":xylene}
        
        data = pd.read_csv('static/city_day.csv')

        # Remove rows with missing values
        data = data.dropna()

        # Drop the 'City' and 'Date' columns
        data = data.drop(['City', 'Date'], axis=1)

        # Select the features and target variable
        features = data.drop(['AQI_Bucket', 'AQI'], axis=1)
        target = data['AQI_Bucket']

        # Convert categorical variables to numerical using one-hot encoding
        features = pd.get_dummies(features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Create a random forest classifier model  (algorithm) 
        model = RandomForestClassifier()
        # multiple independent variable or data or column

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Save the trained model using pickle
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Load the saved model using pickle
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        # pickel to take custome input

        # Get user input for a single test case
        input_data = []
        columns = list(features.columns)
        for column in columns:
            if column not in ['City', 'Date']:
                # value = input(f"Enter the value for {column}: ")
                
                for x,y in datainfo.items():
                    if column == x:    
                        value=y
                if value.strip() == '':
                        # Replace missing value with NaN
                    input_data.append(np.nan)
                else:
                    input_data.append(float(value))

        # Make predictions on the user input data
        test_case = pd.DataFrame([input_data], columns=columns)

        # Fill missing values with the mean of the respective column
        test_case = test_case.fillna(X_test.mean())

        predicted_aqi_bucket = model.predict(test_case)

        print("Predicted AQI Bucket:", predicted_aqi_bucket)
        finaldata={'Accuracy': accuracy,'prediction':predicted_aqi_bucket}

        return render_template("answer.html",finaldata=finaldata)
    else:
        return "Bad Request"

@app.route('/')
def hello_world():
    # Load the data into a pandas DataFrame

    return render_template("homepage.html")

if __name__ == '__main__':
    app.run(debug=True)