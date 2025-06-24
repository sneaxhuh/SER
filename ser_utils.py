import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf

# Constants
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
SEGMENT_START = 0.5
SEGMENT_END = 3.5
MFCC_TARGET_LEN = 184

EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
LABEL_MAP = {i: e for i, e in enumerate(EMOTION_LABELS)}
NUM_CLASSES = len(EMOTION_LABELS)
LSTM_HIDDEN_SIZE = 128

class CRNN_SER(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(input_size=256 * 2, hidden_size=LSTM_HIDDEN_SIZE,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)  # Supports all formats if ffmpeg installed
        y = y[int(SEGMENT_START * SAMPLE_RATE):int(SEGMENT_END * SAMPLE_RATE)]
        required_len = int((SEGMENT_END - SEGMENT_START) * SAMPLE_RATE)
        if len(y) < required_len:
            y = np.pad(y, (0, required_len - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        return mfcc.T
    except Exception as e:
        print(f"[Error loading audio]: {file_path} - {e}")
        return None

def pad_or_truncate(mfcc, target_len=MFCC_TARGET_LEN):
    mfcc = torch.tensor(mfcc, dtype=torch.float32)
    if mfcc.shape[0] > target_len:
        return mfcc[:target_len]
    elif mfcc.shape[0] < target_len:
        return torch.cat([mfcc, torch.zeros((target_len - mfcc.shape[0], mfcc.shape[1]))])
    return mfcc

def predict(model, mfcc_tensor, device):
    X = mfcc_tensor.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return LABEL_MAP[pred] # Return both prediction and confidence scores
