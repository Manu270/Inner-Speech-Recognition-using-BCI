from flask import Flask, request, send_file
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix for macOS GUI crash
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and label encoder
model = load_model('eegnet_model.h5')
le = joblib.load('label_encoder.pkl')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EEGNet Inner Speech Predictor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <style>
    html, body {
      height: 100%;
      background: linear-gradient(to bottom, #e3f2fd,rgb(200, 225, 230));
      font-family: 'Segoe UI', sans-serif;
    }
    .navbar {
      background-color:rgb(46, 121, 125);
    }
    .navbar-brand {
      color: white;
      font-weight: bold;
      font-size: 1.3rem;
    }
    .container {
      max-width: 900px;
      background-color: #ffffffee;
      border-radius: 20px;
      padding: 30px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
      margin-top: 40px;
    }
    .result-box {
      background-color:rgb(233, 242, 248);
      border-left: 5px solidrgb(67, 149, 160);
      padding: 25px;
      margin-top: 20px;
      border-radius: 10px;
    }
    img {
      max-width: 100%;
      border-radius: 10px;
      margin-top: 15px;
    }
    .btn-primary {
      background-color:rgb(56, 116, 142);
      border: none;
    }
    .btn-primary:hover {
      background-color:rgb(46, 109, 125);
    }
    .spinner-border {
      display: none;
    }
  </style>
  <script>
    function showSpinner() {
      document.getElementById('spinner').style.display = 'inline-block';
    }
  </script>
</head>
<body>
  <nav class="navbar navbar-expand-lg">
    <div class="container-fluid justify-content-center">
      <a class="navbar-brand text-center" href="#"> EEGNet Inner Speech Predictor </a>
    </div>
  </nav>
  <div class="container text-center">
    {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
    {% if prediction %}
    <div class="result-box">
      <h5><strong>Predicted Sentence:</strong></h5>
      <p class="text-success fs-4">{{ prediction }}</p>
      <p class="mt-3"><strong>File:</strong> {{ filename }}</p>
      {% if plots %}
      <ul class="nav nav-tabs" id="myTab" role="tablist">
        {% for title, _ in plots.items() %}
        <li class="nav-item" role="presentation">
          <button class="nav-link {% if loop.first %}active{% endif %}" id="tab-{{ loop.index }}" data-bs-toggle="tab" data-bs-target="#plot-{{ loop.index }}" type="button" role="tab">{{ title }}</button>
        </li>
        {% endfor %}
      </ul>
      <div class="tab-content mt-3">
        {% for title, data in plots.items() %}
        <div class="tab-pane fade {% if loop.first %}show active{% endif %}" id="plot-{{ loop.index }}" role="tabpanel">
          <img src="data:image/png;base64,{{ data }}">
        </div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endif %}
    <form method="post" enctype="multipart/form-data" onsubmit="showSpinner()">
      <div class="mb-3 mt-4">
        <label class="form-label" title="Upload a .csv EEG file with shape (8, 256)">Choose EEG CSV File:</label>
        <input class="form-control" type="file" name="file" accept=".csv" required>
      </div>
      <button class="btn btn-primary" type="submit">Upload & Predict</button>
      <div id="spinner" class="spinner-border text-success ms-3" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
    </form>
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    from flask import render_template_string
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="No file selected.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            eeg_data = pd.read_csv(filepath, header=None).values
            if eeg_data.shape != (8, 256):
                return render_template_string(HTML_TEMPLATE, error="Invalid shape. Expected (8, 256).")

            eeg_input = eeg_data.reshape(1, 1, 8, 256)
            pred = model.predict(eeg_input)
            predicted_sentence = le.classes_[np.argmax(pred)]

            def plot_to_base64(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode('utf-8')

            plots = {}

            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(np.corrcoef(eeg_data), annot=True, cmap='coolwarm', ax=ax1,
                        xticklabels=[f'Ch{i+1}' for i in range(8)], yticklabels=[f'Ch{i+1}' for i in range(8)])
            ax1.set_title("EEG Channel Correlation")
            plots["Correlation"] = plot_to_base64(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            for ch in range(8):
                ax2.plot(eeg_data[ch], label=f'Ch {ch+1}')
            ax2.set_title("Raw EEG Signal")
            ax2.legend()
            plots["Raw Signal"] = plot_to_base64(fig2)

            from scipy.signal import welch
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            for ch in range(8):
                freqs, psd = welch(eeg_data[ch], fs=128, nperseg=128)
                ax3.semilogy(freqs, psd, label=f'Ch {ch+1}')
            ax3.set_title("Power Spectral Density (PSD)")
            ax3.legend()
            plots["PSD"] = plot_to_base64(fig3)

            fig4, ax4 = plt.subplots(figsize=(8, 4))
            mean_signal = eeg_data.mean(axis=0)
            ax4.plot(mean_signal)
            ax4.set_title("Mean EEG Signal (All Channels)")
            plots["Mean Signal"] = plot_to_base64(fig4)

            return render_template_string(HTML_TEMPLATE, prediction=predicted_sentence, filename=file.filename, plots=plots)

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🚀 Flask server started! Visit http://localhost:5050")
    app.run(debug=True, host='0.0.0.0', port=5050)



# from flask import Flask, request, send_file
# import os
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Fix for macOS GUI crash
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import io
# import base64

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load model and label encoder
# model = load_model('eegnet_model.h5')
# le = joblib.load('label_encoder.pkl')

# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8">
#   <title>EEGNet Inner Speech Predictor</title>
#   <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
#   <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
#   <style>
#     html, body {
#       height: 100%;
#       background: linear-gradient(to bottom, #f3e5f5, #e1f5fe);
#       font-family: 'Segoe UI', sans-serif;
#     }
#     .navbar {
#       background-color: #6a1b9a;
#     }
#     .navbar-brand {
#       color: white;
#       font-weight: bold;
#     }
#     .container {
#       max-width: 860px;
#       background-color: white;
#       border-radius: 15px;
#       padding: 30px;
#       box-shadow: 0 8px 20px rgba(0,0,0,0.15);
#       margin-top: 40px;
#     }
#     .result-box {
#       background-color: #f3fbe7;
#       border-left: 6px solid #388e3c;
#       padding: 20px;
#       margin-top: 20px;
#       border-radius: 10px;
#     }
#     img {
#       max-width: 100%;
#       border-radius: 10px;
#       margin-top: 15px;
#     }
#     .btn-primary {
#       background-color: #6a1b9a;
#       border: none;
#     }
#     .btn-primary:hover {
#       background-color: #4a148c;
#     }
#     .spinner-border {
#       display: none;
#     }
#   </style>
#   <script>
#     function showSpinner() {
#       document.getElementById('spinner').style.display = 'inline-block';
#     }
#   </script>
# </head>
# <body>
#   <nav class="navbar navbar-expand-lg">
#     <div class="container-fluid">
#       <a class="navbar-brand mx-auto" href="#">🧠 EEGNet - Inner Speech Predictor</a>
#     </div>
#   </nav>
#   <div class="container text-center">
#     {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
#     {% if prediction %}
#     <div class="result-box">
#       <h5><strong>Predicted Sentence:</strong></h5>
#       <p class="text-success fs-4">{{ prediction }}</p>
#       <p class="mt-3"><strong>File:</strong> {{ filename }}</p>
#       {% if plots %}
#       <ul class="nav nav-tabs" id="myTab" role="tablist">
#         {% for title, _ in plots.items() %}
#         <li class="nav-item" role="presentation">
#           <button class="nav-link {% if loop.first %}active{% endif %}" id="tab-{{ loop.index }}" data-bs-toggle="tab" data-bs-target="#plot-{{ loop.index }}" type="button" role="tab">{{ title }}</button>
#         </li>
#         {% endfor %}
#       </ul>
#       <div class="tab-content mt-3">
#         {% for title, data in plots.items() %}
#         <div class="tab-pane fade {% if loop.first %}show active{% endif %}" id="plot-{{ loop.index }}" role="tabpanel">
#           <img src="data:image/png;base64,{{ data }}">
#         </div>
#         {% endfor %}
#       </div>
#       {% endif %}
#     </div>
#     {% endif %}
#     <form method="post" enctype="multipart/form-data" onsubmit="showSpinner()">
#       <div class="mb-3 mt-4">
#         <label class="form-label" title="Upload a .csv EEG file with shape (8, 256)">Choose EEG CSV File:</label>
#         <input class="form-control" type="file" name="file" accept=".csv" required>
#       </div>
#       <button class="btn btn-primary" type="submit">Upload & Predict</button>
#       <div id="spinner" class="spinner-border text-primary ms-3" role="status">
#         <span class="visually-hidden">Loading...</span>
#       </div>
#     </form>
#   </div>
# </body>
# </html>
# '''

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     from flask import render_template_string
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template_string(HTML_TEMPLATE, error="No file uploaded.")

#         file = request.files['file']
#         if file.filename == '':
#             return render_template_string(HTML_TEMPLATE, error="No file selected.")

#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         try:
#             eeg_data = pd.read_csv(filepath, header=None).values
#             if eeg_data.shape != (8, 256):
#                 return render_template_string(HTML_TEMPLATE, error="Invalid shape. Expected (8, 256).")

#             eeg_input = eeg_data.reshape(1, 1, 8, 256)
#             pred = model.predict(eeg_input)
#             predicted_sentence = le.classes_[np.argmax(pred)]

#             def plot_to_base64(fig):
#                 buf = io.BytesIO()
#                 fig.savefig(buf, format='png')
#                 plt.close(fig)
#                 buf.seek(0)
#                 return base64.b64encode(buf.read()).decode('utf-8')

#             plots = {}

#             fig1, ax1 = plt.subplots(figsize=(6, 5))
#             sns.heatmap(np.corrcoef(eeg_data), annot=True, cmap='coolwarm', ax=ax1,
#                         xticklabels=[f'Ch{i+1}' for i in range(8)], yticklabels=[f'Ch{i+1}' for i in range(8)])
#             ax1.set_title("EEG Channel Correlation")
#             plots["Correlation"] = plot_to_base64(fig1)

#             fig2, ax2 = plt.subplots(figsize=(8, 4))
#             for ch in range(8):
#                 ax2.plot(eeg_data[ch], label=f'Ch {ch+1}')
#             ax2.set_title("Raw EEG Signal")
#             ax2.legend()
#             plots["Raw Signal"] = plot_to_base64(fig2)

#             from scipy.signal import welch
#             fig3, ax3 = plt.subplots(figsize=(8, 4))
#             for ch in range(8):
#                 freqs, psd = welch(eeg_data[ch], fs=128, nperseg=128)
#                 ax3.semilogy(freqs, psd, label=f'Ch {ch+1}')
#             ax3.set_title("Power Spectral Density (PSD)")
#             ax3.legend()
#             plots["PSD"] = plot_to_base64(fig3)

#             fig4, ax4 = plt.subplots(figsize=(8, 4))
#             mean_signal = eeg_data.mean(axis=0)
#             ax4.plot(mean_signal)
#             ax4.set_title("Mean EEG Signal (All Channels)")
#             plots["Mean Signal"] = plot_to_base64(fig4)

#             return render_template_string(HTML_TEMPLATE, prediction=predicted_sentence, filename=file.filename, plots=plots)

#         except Exception as e:
#             return render_template_string(HTML_TEMPLATE, error=str(e))

#     return render_template_string(HTML_TEMPLATE)

# if __name__ == '__main__':
#     print("🚀 Flask server started! Visit http://localhost:5050")
#     app.run(debug=True, host='0.0.0.0', port=5050)




# from flask import Flask, request, send_file
# import os
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Fix for macOS GUI crash
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import io
# import base64

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load model and label encoder
# model = load_model('eegnet_model.h5')
# le = joblib.load('label_encoder.pkl')

# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8">
#   <title>EEGNet Inner Speech Predictor</title>
#   <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
#   <style>
#     html, body {
#       height: 100%;
#     }
#     body {
#       margin: 0;
#       padding-top: 60px;
#       background: linear-gradient(180deg, #c1f0f6, #d3cce3);
#       font-family: 'Segoe UI', sans-serif;
#       display: flex;
#       justify-content: center;
#       align-items: flex-start;
#     }
#     .container {
#       width: 90%;
#       max-width: 750px;
#       background-color: white;
#       border-radius: 10px;
#       padding: 30px;
#       box-shadow: 0 0 25px rgba(0, 0, 0, 0.1);
#     }
#     .result-box {
#       background-color: #f1f8e9;
#       border-left: 6px solid #4caf50;
#       padding: 20px;
#       margin-top: 20px;
#       border-radius: 5px;
#     }
#     img {
#       max-width: 100%;
#       border-radius: 10px;
#       margin-top: 15px;
#     }
#     .btn-primary {
#       background-color: #7b1fa2;
#       border: none;
#     }
#     .btn-primary:hover {
#       background-color: #6a1b9a;
#     }
#     h2 {
#       color: #4a148c;
#     }
#   </style>
# </head>
# <body>
#   <div class="container text-center">
#     <h2 class="mb-4">🧠 EEGNet Inner Speech Classifier</h2>
#     {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
#     {% if prediction %}
#     <div class="result-box">
#       <h5><strong>Predicted Sentence:</strong></h5>
#       <p class="text-success fs-4">{{ prediction }}</p>
#       <p><strong>File:</strong> {{ filename }}</p>
#       {% if image_data %}
#       <h6>Channel Correlation Heatmap</h6>
#       <img src="data:image/png;base64,{{ image_data }}">
#       {% endif %}
#     </div>
#     {% endif %}
#     <form method="post" enctype="multipart/form-data">
#       <div class="mb-3">
#         <input class="form-control" type="file" name="file" accept=".csv" required>
#       </div>
#       <button class="btn btn-primary" type="submit">Upload & Predict</button>
#     </form>
#   </div>
# </body>
# </html>
# '''

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     from flask import render_template_string
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template_string(HTML_TEMPLATE, error="No file uploaded.")

#         file = request.files['file']
#         if file.filename == '':
#             return render_template_string(HTML_TEMPLATE, error="No file selected.")

#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         try:
#             eeg_data = pd.read_csv(filepath, header=None).values
#             if eeg_data.shape != (8, 256):
#                 return render_template_string(HTML_TEMPLATE, error="Invalid shape. Expected (8, 256).")

#             eeg_input = eeg_data.reshape(1, 1, 8, 256)
#             pred = model.predict(eeg_input)
#             predicted_sentence = le.classes_[np.argmax(pred)]

#             # Visualization: EEG channel correlation heatmap
#             fig, ax = plt.subplots(figsize=(6, 5))
#             corr_matrix = np.corrcoef(eeg_data)
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax,
#                         xticklabels=[f'Ch{i+1}' for i in range(8)],
#                         yticklabels=[f'Ch{i+1}' for i in range(8)])
#             ax.set_title("EEG Channel Correlation")

#             buf = io.BytesIO()
#             plt.savefig(buf, format='png')
#             plt.close(fig)
#             buf.seek(0)
#             image_data = base64.b64encode(buf.read()).decode('utf-8')

#             return render_template_string(HTML_TEMPLATE, prediction=predicted_sentence, filename=file.filename, image_data=image_data)

#         except Exception as e:
#             return render_template_string(HTML_TEMPLATE, error=str(e))

#     return render_template_string(HTML_TEMPLATE)

# if __name__ == '__main__':
#     print("🚀 Flask server started! Visit http://localhost:5050")
#     app.run(debug=True, host='0.0.0.0', port=5050)
