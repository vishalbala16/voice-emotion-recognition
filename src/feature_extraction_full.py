import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from scipy import signal
from scipy.fft import fft

class FeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_mfcc_simple(self, audio, n_mfcc=13):
        """Simple MFCC implementation without librosa"""
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Framing
        frame_size = 0.025
        frame_stride = 0.01
        frame_length = int(round(frame_size * self.sr))
        frame_step = int(round(frame_stride * self.sr))
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                 np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # Windowing
        frames *= np.hamming(frame_length)
        
        # FFT and Power Spectrum
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
        
        # Filter Banks
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sr / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((NFFT + 1) * hz_points / self.sr)
        
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])
            
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        # DCT
        mfcc = np.zeros((num_frames, n_mfcc))
        for i in range(n_mfcc):
            mfcc[:, i] = np.sum(filter_banks * np.cos(np.pi * i * (2 * np.arange(nfilt) + 1) / (2 * nfilt)), axis=1)
        
        return np.mean(mfcc, axis=0)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        # FFT
        fft_spectrum = np.abs(fft(audio))
        magnitude = fft_spectrum[:len(fft_spectrum)//2]
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(audio), 1/self.sr)[:len(magnitude)]
        
        # Spectral Centroid
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Spectral Rolloff (85% of energy)
        cumsum_mag = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum_mag >= 0.85 * cumsum_mag[-1])[0][0]
        spectral_rolloff = freqs[rolloff_idx]
        
        # Spectral Bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        
        # Zero Crossing Rate
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        zero_crossing_rate = len(zero_crossings) / len(audio)
        
        return np.array([spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate])
    
    def extract_chroma_simple(self, audio):
        """Simple chroma feature extraction"""
        # Simple pitch class profile
        fft_spectrum = np.abs(fft(audio))
        magnitude = fft_spectrum[:len(fft_spectrum)//2]
        freqs = np.fft.fftfreq(len(audio), 1/self.sr)[:len(magnitude)]
        
        # Map frequencies to pitch classes (12 semitones)
        chroma = np.zeros(12)
        for i, freq in enumerate(freqs):
            if freq > 0:
                # Convert frequency to MIDI note number
                midi_note = 69 + 12 * np.log2(freq / 440.0)
                pitch_class = int(midi_note) % 12
                chroma[pitch_class] += magnitude[i]
        
        # Normalize
        if np.sum(chroma) > 0:
            chroma = chroma / np.sum(chroma)
        
        return chroma
    
    def extract_mel_spectrogram(self, audio, n_mels=128):
        """Extract mel-spectrogram for deep learning"""
        # STFT
        f, t, Zxx = signal.stft(audio, self.sr, nperseg=512, noverlap=256)
        magnitude = np.abs(Zxx)
        
        # Mel filter bank
        n_fft = 512
        fmin = 0
        fmax = self.sr / 2
        
        # Create mel scale
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        # Convert to FFT bin numbers
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)
        
        # Create filter bank
        fbank = np.zeros((n_mels, int(n_fft // 2 + 1)))
        for m in range(1, n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]
            
            for k in range(left, center):
                if center != left:
                    fbank[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right != center:
                    fbank[m - 1, k] = (right - k) / (right - center)
        
        # Apply mel filter bank
        mel_spec = np.dot(fbank, magnitude)
        
        # Convert to log scale
        mel_spec = np.log(mel_spec + 1e-10)
        
        return mel_spec
    
    def extract_all_features(self, audio):
        """Extract all features for classical ML"""
        mfcc = self.extract_mfcc_simple(audio)
        chroma = self.extract_chroma_simple(audio)
        spectral = self.extract_spectral_features(audio)
        
        return np.concatenate([mfcc, chroma, spectral])

def main():
    # Load processed audio data
    try:
        audio_data = np.load('../data/processed_audio.npy')
        labels = np.load('../data/labels.npy')
    except FileNotFoundError:
        print("Processed audio data not found. Run audio_preprocessing.py first.")
        return
    
    extractor = FeatureExtractor()
    
    print("Extracting features...")
    
    # Extract features for classical ML
    classical_features = []
    spectrograms = []
    
    for i, audio in enumerate(audio_data):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(audio_data)}")
        
        # Classical features
        features = extractor.extract_all_features(audio)
        classical_features.append(features)
        
        # Mel-spectrogram for deep learning
        mel_spec = extractor.extract_mel_spectrogram(audio)
        spectrograms.append(mel_spec)
    
    classical_features = np.array(classical_features)
    spectrograms = np.array(spectrograms)
    
    # Normalize classical features
    scaler = StandardScaler()
    classical_features_scaled = scaler.fit_transform(classical_features)
    
    # Save features
    np.save('../data/classical_features.npy', classical_features_scaled)
    np.save('../data/spectrograms.npy', spectrograms)
    joblib.dump(scaler, '../models/feature_scaler.pkl')
    
    print(f"Classical features shape: {classical_features_scaled.shape}")
    print(f"Spectrograms shape: {spectrograms.shape}")
    print("Feature extraction completed!")

if __name__ == "__main__":
    main()