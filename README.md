# 🎧 Auralsense: Real-Time Speech Emotion Recognition with CRNN

AuralSense is a full-stack web application that predicts human emotions from speech using a Convolutional Recurrent Neural Network (CRNN). It supports real-time emotion prediction on audio files and provides a clean UI with waveform visualizations and results.
---

## 🧪 Preprocessing Methodology

Each audio sample is:

1. **Resampled** to 16 kHz mono
2. **Trimmed** to a fixed segment (0.5s to 3.5s)
3. **Converted** into 40-dimensional MFCCs using Librosa
4. **Standardized** (zero-mean, unit variance)
5. **Padded/Truncated** to a fixed length of 184 time steps

This results in a consistent input shape of **(1, 1, 184, 40)** for each file.

---

## 🧠 Model Pipeline

The model architecture is a CRNN:
- **CNN layers** extract time-frequency spatial features
- **Bi-LSTM layers** model temporal dependencies
- **Fully connected layer** classifies into one of 8 emotions

The model was trained using the **RAVDESS** dataset, covering the following classes:

- `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

---

## 📊 Evaluation

### ✅ Classification Report (on test set):

| Emotion    | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Angry      | 0.855     | 0.934  | 0.893    | 76.0    |
| Calm       | 0.861     | 0.895  | 0.877    | 76.0    |
| Disgust    | 0.829     | 0.872  | 0.850    | 39.0    |
| Fearful    | 0.831     | 0.776  | 0.803    | 76.0    |
| Happy      | 0.914     | 0.697  | 0.791    | 76.0    |
| Neutral    | 0.756     | 0.872  | 0.810    | 39.0    |
| Sad        | 0.848     | 0.882  | 0.865    | 76.0    |
| Surprised  | 0.829     | 0.872  | 0.850    | 39.0    |
| **Macro Avg** | **0.840** | **0.850** | **0.842** |         |
| **Weighted Avg** | **0.842** | **0.844** | **0.843** |         |


---

## 🔧 Usage — Running `predict.py`

You can use `predict.py` to analyze emotions from `.wav` audio either on a **single file** or a **folder** of files.

---

### 📁 Folder Mode

To analyze all `.wav` files in a folder and save predictions in a CSV file:

```bash
python3 predict.py --folder ./audio_samples/ --model crnn.pth --output <FILE_NAME>.csv
```
✅ This will:

Load all audio files inside ./audio_samples/

Predict emotion for each file using the CRNN model

Save a CSV named <FILE_NAME> in the same folder with the results


### 🎧 Single File Mode

To analyze a single .wav audio file:

```bash
python3 predict.py --file ./audio_samples/happy_02.wav --model crnn.pth
```
✅ This will:

Predict the emotion for happy_02.wav

Output the result on the terminal
