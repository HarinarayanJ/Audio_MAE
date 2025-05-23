import os
import librosa
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# SETTINGS
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160  # (1024 / 160 = 64 frames per second)
N_FFT = 400
TARGET_NUM_FRAMES = 1024
AUDIO_DIR = "C:/Users/harin/Hari/ms/ESC-50-master/audio"
SAVE_DIR = "mel"

os.makedirs(SAVE_DIR, exist_ok=True)

# Create MelSpectrogram transform
mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
db_transform = AmplitudeToDB()

def load_and_convert(file_path):
    # Load and resample
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad or truncate to fit exactly 1024 frames
    mel = mel_transform(waveform)
    mel_db = db_transform(mel)
    mel_db = mel_db[:, :, :TARGET_NUM_FRAMES] if mel_db.shape[-1] > TARGET_NUM_FRAMES else \
             torch.nn.functional.pad(mel_db, (0, TARGET_NUM_FRAMES - mel_db.shape[-1]))

    return mel_db  # [1, 128, 1024]

def process_all():
    for file_name in tqdm(os.listdir(AUDIO_DIR)):
        if not file_name.endswith('.wav'):
            continue
        full_path = os.path.join(AUDIO_DIR, file_name)
        mel = load_and_convert(full_path)  # [1, 128, 1024]
        mel = mel.permute(0, 2, 1)  # -> [1, 1024, 128]
        torch.save(mel, os.path.join(SAVE_DIR, file_name.replace('.wav', '.pt')))
        print(f"Processed {file_name} to {file_name.replace('.wav', '.pt')}")

if __name__ == "__main__":
    process_all()
    print("All files processed and saved.")