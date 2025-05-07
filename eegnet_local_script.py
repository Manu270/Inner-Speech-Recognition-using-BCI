
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense)
import mne

# === CONFIGURATION ===
DATASET_PATH = './inner_speech_25_sentences.csv'
SINGLE_EEG_FILE_PATH = './sample_eeg_signal.csv'  # EEG signal for prediction

# === Load Dataset ===
df = pd.read_csv(DATASET_PATH)
X = df.drop('label', axis=1).values
y = df['label'].values

# === Preprocessing ===
X = X.reshape(X.shape[0], 8, 256).transpose(0, 1, 2)
X = X[:, np.newaxis, :, :]  # Shape: (samples, 1, channels, time)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# === EEGNet Model ===
def EEGNet(nb_classes, Chans=8, Samples=256, dropoutRate=0.5):
    input1 = Input(shape=(1, Chans, Samples))

    block1 = Conv2D(16, (1, 64), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2, padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, activation='softmax')(flatten)
    return Model(inputs=input1, outputs=dense)

# === Train Model ===
model = EEGNet(nb_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# === Evaluate ===
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
print(classification_report(y_true, y_pred_class, target_names=le.classes_))



# === Save Model and Label Encoder ===
from tensorflow.keras.models import save_model
import joblib

# Save trained EEGNet model
model.save("eegnet_model.h5")
print("✅ Model saved as eegnet_model.h5")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")
print("✅ Label encoder saved as label_encoder.pkl")





# === Predict on Single EEG Signal ===
eeg_data = pd.read_csv(SINGLE_EEG_FILE_PATH, header=None).values
if eeg_data.shape != (8, 256):
    raise ValueError(f"Expected shape (8, 256), but got {eeg_data.shape}")

eeg_input = eeg_data.reshape(1, 1, 8, 256)
pred = model.predict(eeg_input)
predicted_sentence = le.classes_[np.argmax(pred)]
print(f"✅ Predicted inner speech sentence: '{predicted_sentence}'")

# === Visualizations ===
X_vis = df.drop('label', axis=1).values.reshape(len(df), 8, 256)

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Sentence Class Distribution")
plt.xticks(rotation=15)
plt.grid(True)
plt.show()

# Raw EEG signal from 1 sample per class
labels = np.unique(y)
plt.figure(figsize=(15, 8))
for i, label in enumerate(labels):
    sample = X_vis[np.where(y == label)[0][0]]
    plt.subplot(len(labels), 1, i+1)
    for ch in range(8):
        plt.plot(sample[ch], label=f'Ch {ch+1}')
    plt.title(f"Raw EEG - {label}")
    plt.xlabel("Timepoints")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right', ncol=4, fontsize=8)
    plt.tight_layout()
plt.show()

# Mean EEG signal per sentence
plt.figure(figsize=(10, 6))
for label in labels:
    class_data = X_vis[np.where(y == label)]
    mean_signal = class_data.mean(axis=(0, 1))
    plt.plot(mean_signal, label=label)
plt.title("Mean EEG Signal per Sentence")
plt.xlabel("Timepoints")
plt.ylabel("Mean Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Power Spectral Density for one sample per class
plt.figure(figsize=(14, 8))
for i, label in enumerate(labels):
    sample = X_vis[np.where(y == label)[0][0]]
    plt.subplot(len(labels), 1, i+1)
    for ch in range(8):
        freqs, psd = welch(sample[ch], fs=128, nperseg=128)
        plt.semilogy(freqs, psd, label=f'Ch {ch+1}')
    plt.title(f"PSD - {label}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend(loc='upper right', ncol=4, fontsize=8)
    plt.tight_layout()
plt.show()

# EEG Channel Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = np.corrcoef(X_vis[0])
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            xticklabels=[f'Ch{i+1}' for i in range(8)],
            yticklabels=[f'Ch{i+1}' for i in range(8)])
plt.title("EEG Channel Correlation (Sample 1)")
plt.show()

# === Topographic Map using MNE ===
data = X_vis[0]
standard_names = ['Fp1', 'Fz', 'Cz', 'Pz', 'O1', 'O2', 'C3', 'C4']
sfreq = 128
info = mne.create_info(ch_names=standard_names, sfreq=sfreq, ch_types=['eeg']*8)
raw = mne.io.RawArray(data, info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

num_segments = 4
segment_size = raw.n_times // num_segments

for i in range(num_segments):
    start = i * segment_size
    end = (i + 1) * segment_size if i < num_segments - 1 else raw.n_times
    mean_data = np.mean(raw.get_data()[:, start:end], axis=1)

    mne.viz.plot_topomap(
        mean_data,
        raw.info,
        show=True,
        contours=True,
        res=300,
        sensors='k.',
        size=5,
        vlim=(-1e-5, 1e-5),
        cmap='RdBu_r'
    )


# === Train Model ===
model = EEGNet(nb_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# === Save Model and Label Encoder ===
model.save("eegnet_model.h5")
print("✅ Model saved as eegnet_model.h5")

joblib.dump(le, "label_encoder.pkl")
print("✅ Label encoder saved as label_encoder.pkl")

# === Evaluate ===
# (rest of your code continues as is...)
