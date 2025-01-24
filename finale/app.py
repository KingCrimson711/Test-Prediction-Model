from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input values from the form
            hours_studied = float(request.form['hours_studied'])
            previous_scores = float(request.form['previous_scores'])
            extracurricular_activities = 1 if request.form['extracurricular_activities'] == 'Yes' else 0
            sleep_hours = float(request.form['sleep_hours'])
            question_papers_practiced = float(request.form['question_papers_practiced'])

            # Debug: Print received values
            print(f"Received values: hours_studied={hours_studied}, previous_scores={previous_scores}, "
                  f"extracurricular_activities={extracurricular_activities}, sleep_hours={sleep_hours}, "
                  f"question_papers_practiced={question_papers_practiced}")

            # Prepare the data for prediction
            input_data = np.array([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, question_papers_practiced]])

            # Normalize the input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data)

            # Debug: Print the scaled input data
            print(f"Scaled input data: {input_data_scaled}")

            # Make the prediction
            prediction = model.predict(input_data_scaled)[0][0]

            # Debug: Print the prediction
            print(f"Prediction: {prediction}")
        except Exception as e:
            print(f"Error occurred: {e}")
            prediction = "Error occurred during prediction. Check console for details."
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
