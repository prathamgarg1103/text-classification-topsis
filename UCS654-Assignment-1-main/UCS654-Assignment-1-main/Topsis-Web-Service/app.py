import os
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SENDER_EMAIL = "rbrar_be23@thapar.edu"
SENDER_PASSWORD = os.environ.get("EMAIL_PASSWORD")

def calculate_topsis(file_path, weights, impacts):
    df = pd.read_csv(file_path)
    temp_df = df.iloc[:, 1:].values.astype(float)
    weights = [float(w) for w in weights.split(',')]
    impacts = impacts.split(',')

    rss = np.sqrt(np.sum(temp_df**2, axis=0))
    normalized_matrix = temp_df / rss
    weighted_matrix = normalized_matrix * weights
    
    ideal_best = []
    ideal_worst = []
    
    for i in range(len(weights)):
        if impacts[i] == '+':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))
            
    s_plus = np.sqrt(np.sum((weighted_matrix - np.array(ideal_best))**2, axis=1))
    s_minus = np.sqrt(np.sum((weighted_matrix - np.array(ideal_worst))**2, axis=1))
    performance_score = s_minus / (s_plus + s_minus)
    
    df['Topsis Score'] = performance_score
    df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
    df.to_csv(output_path, index=False)
    return output_path

def send_email(receiver_email, result_file):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = "TOPSIS Result File"
    msg.attach(MIMEText("Here is your TOPSIS result file.", 'plain'))

    attachment = open(result_file, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename=result.csv")
    msg.attach(part)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    text = msg.as_string()
    server.sendmail(SENDER_EMAIL, receiver_email, text)
    server.quit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            weights = request.form['weights']
            impacts = request.form['impacts']
            email = request.form['email']

            if not file: return render_template('index.html', message="No file uploaded", status="error")
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            w_count = len(weights.split(','))
            i_count = len(impacts.split(','))
            df = pd.read_csv(file_path)
            c_count = len(df.columns) - 1 
            
            if w_count != i_count or w_count != c_count:
                return render_template('index.html', message=f"Error: Column count ({c_count}), Weights ({w_count}), and Impacts ({i_count}) must match.", status="error")

            result_path = calculate_topsis(file_path, weights, impacts)
            send_email(email, result_path)

            return render_template('index.html', message="Success! Result sent to email.", status="success")

        except Exception as e:
            return render_template('index.html', message=f"Error: {str(e)}", status="error")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)