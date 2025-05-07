<img width="1465" alt="Screenshot 2025-05-07 at 11 09 47 AM" src="https://github.com/user-attachments/assets/4af3d161-bf43-458a-bcfe-0233fbe5e5ae" />
<img width="1469" alt="Screenshot 2025-05-07 at 11 09 15 AM" src="https://github.com/user-attachments/assets/48f3a88a-10a9-46fa-b57e-6fc2b626f3f7" />
<img width="1467" alt="Screenshot 2025-05-07 at 11 08 20 AM" src="https://github.com/user-attachments/assets/6abf0a81-d8d8-4ddd-8fb1-60b7cfb0c139" />
# 🧠 Inner Speech Classification using EEGNet

This project implements an end-to-end pipeline to classify **inner speech EEG signals** using the EEGNet deep learning model. It includes data preprocessing, training, evaluation, prediction, and various visualizations like PSD plots, topoplots, and correlation heatmaps.

## 📁 Repository Structure

```

├── model/                   # EEGNet model architecture
├── data/                    # EEG dataset (CSV format)
├── scripts/
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation script
│   ├── predict.py           # Predict single EEG input
│   └── visualize.py         # EEG signal visualizations
├── app/
│   └── flask\_app.py         # Web app interface using Flask
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── inner\_speech\_25\_sentences.csv # Sample dataset

````

---

## 🚀 Features

- EEGNet architecture (optimized for EEG data)
- Train/test split and one-hot encoding
- Visualizations:
  - Class distribution
  - Raw EEG signal per sentence
  - Mean EEG signal across time
  - Power Spectral Density (PSD)
  - EEG channel correlation heatmap
  - EEG topographic maps (topoplots) using MNE
- Live sentence prediction from uploaded EEG signals
- Flask web interface for interactive prediction

---

## 🧪 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy / Pandas / Matplotlib / Seaborn
- MNE-Python
- Flask
- scikit-learn

---

## 📦 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/eegnet-inner-speech.git
   cd eegnet-inner-speech
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run training:

   ```bash
   python scripts/train.py
   ```

4. Evaluate the model:

   ```bash
   python scripts/evaluate.py
   ```

5. Predict a sentence from new EEG signal:

   ```bash
   python scripts/predict.py --file path/to/eeg_sample.csv
   ```

6. Launch the web app:

   ```bash
   cd app
   python flask_app.py
   ```

---

## 🧠 Dataset

* Format: `.csv`
* Shape: `(samples, 2048 features + 1 label column)`
* Reshaped to: `(samples, 8 channels, 256 time points)`
* Each row represents a trial of inner speech for one of 25 sentence classes.

---

## 📊 Visualizations

You can run `visualize.py` to generate:

* Class distribution plot
* EEG signal per sentence
* Mean EEG signal curve
* PSD plots
* Correlation heatmap
* Topographic brain activity maps (requires `mne`)

---

## ✅ Results

* Classification accuracy: 80% (based on test split)
* Supports real-time prediction on uploaded EEG files
* Demonstrated brainwave differences across inner-speech sentences

<img width="1467" alt="Screenshot 2025-05-07 at 11 08 20 AM" src="https://github.com/user-attachments/assets/0290e888-af0b-4fb1-9b23-ca7f87a396a1" />
<img width="1469" alt="Screenshot 2025-05-07 at 11 09 15 AM" src="https://github.com/user-attachments/assets/7df7899e-74c0-4523-a2a6-32fd0fc5c279" />
<img width="1465" alt="Screenshot 2025-05-07 at 11 09 47 AM" src="https://github.com/user-attachments/assets/439f51b0-5472-447c-b241-ac019791afc4" />





