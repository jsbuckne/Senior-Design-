import os
import torchaudio
import torch
import pandas as pd
from torch.utils.data import Dataset

#Creating a class
class VoiceLineDataset(Dataset):
    
    #import annotations file and audio folder from Dataset:
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    #Defining length
    def __len__(self):
        return len(self.annotations)

    #Loading audio samples and return associated label:
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        #Take signal, sample rate from audio folder path
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr) 
        signal = self._mix_down_if_necessary(signal)
       # If number of samples in signal is more than expected, cut signal and return signal:
        signal = self._cut_if_necessary(signal)
        # If number of samples is less than expected, apply right padding and return signal:
        signal = self._right_pad_if_necessary(signal)
        #passing waveform through function to get mel spectrogram
        signal = self.transformation(signal) 
        return signal, label

    def _cut_if_necessary(self, signal):
        # If signal > expected number of samples, apply multi-dimensional slicing
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        #If signal < expected number of samples, apply right padding
        length_signal = signal.shape[1]
        if length_signal  < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) # [0, 2]:
                # [1, 1, 1] --> [1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    #function to resample audio file if sample rate is not already 16000
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal --> (num_channels, samples) --> (2, 150) --> (1, 150)
        if signal.shape[0] > 1: #(2, 150)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    #Getting file path to audio file from annotations sheet.
    def _get_audio_sample_path(self, index): 
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/jarrett/Desktop/Senior Design/Speech classification /Dataset/Metadata/annotations.csv"
    AUDIO_DIR = "/Users/jarrett/Desktop/Senior Design/Speech classification /Dataset/audio"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000
    device = "cpu"
    print(f"Using device {device}")

   #creating  mel spectrogram:
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE, #set sample rate
        n_fft = 1024, #set frame size
        hop_length = 512, #hop length set to half the frame size
        n_mels = 64 #number of mels
    )
   
    usd = VoiceLineDataset(ANNOTATIONS_FILE,
                           AUDIO_DIR,
                           mel_spectrogram,
                           SAMPLE_RATE,
                           NUM_SAMPLES,
                           device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[1]
