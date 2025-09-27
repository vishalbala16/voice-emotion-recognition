import os
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
import joblib

class AudioPreprocessor:
    def __init__(self, target_sr=16000, duration=3.0):
        self.target_sr = target_sr
        self.duration = duration
        self.max_len = int(target_sr * duration)
    
    def load_and_preprocess(self, file_path):
        """Load and preprocess audio using soundfile (no librosa dependency)"""
        try:
            # Load audio
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Simple resampling (basic decimation/interpolation)
            if sr != self.target_sr:
                ratio = self.target_sr / sr
                if ratio < 1:  # Downsample
                    step = int(1/ratio)
                    audio = audio[::step]
                else:  # Upsample (simple interpolation)
                    audio = np.repeat(audio, int(ratio))
            
            # Trim silence (simple energy-based)
            energy = audio ** 2
            threshold = 0.01 * np.max(energy)
            start_idx = np.argmax(energy > threshold)
            end_idx = len(energy) - np.argmax(energy[::-1] > threshold)
            audio = audio[start_idx:end_idx]
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Pad or truncate to fixed length
            if len(audio) > self.max_len:
                audio = audio[:self.max_len]
            else:
                audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def extract_emotion_from_ravdess(self, filename):
        """Extract emotion from RAVDESS filename format"""
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion_map = {
                '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
            }
            return emotion_map.get(emotion_code, 'unknown')
        return 'unknown'
    
    def process_ravdess_dataset(self, data_dir):
        """Process RAVDESS dataset"""
        audio_data = []
        labels = []
        file_paths = []
        
        if not os.path.exists(data_dir):
            print(f"Dataset directory not found: {data_dir}")
            return np.array([]), np.array([]), []
        
        wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        print(f"Found {len(wav_files)} WAV files")
        
        for file_name in wav_files:
            file_path = os.path.join(data_dir, file_name)
            
            # Extract emotion from filename
            emotion = self.extract_emotion_from_ravdess(file_name)
            if emotion == 'unknown':
                continue
            
            # Process audio
            processed_audio = self.load_and_preprocess(file_path)
            
            if processed_audio is not None:
                audio_data.append(processed_audio)
                labels.append(emotion)
                file_paths.append(file_path)
        
        return np.array(audio_data), np.array(labels), file_paths

def main():
    os.makedirs('../models', exist_ok=True)
    
    preprocessor = AudioPreprocessor()
    
    # Process RAVDESS dataset
    data_dir = '../data/RAVDESS'
    print("Processing RAVDESS dataset...")
    audio_data, labels, file_paths = preprocessor.process_ravdess_dataset(data_dir)
    
    if len(audio_data) == 0:
        print("No audio files processed. Run download_ravdess.py first.")
        return
    
    print(f"Successfully processed {len(audio_data)} audio files")
    print(f"Unique emotions: {np.unique(labels)}")
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Save processed data
    np.save('../data/processed_audio.npy', audio_data)
    np.save('../data/labels.npy', encoded_labels)
    np.save('../data/label_names.npy', labels)
    
    # Save label encoder
    joblib.dump(le, '../models/label_encoder.pkl')
    
    print(f"Audio shape: {audio_data.shape}")
    print(f"Labels: {list(le.classes_)}")
    
    # Create metadata
    metadata = pd.DataFrame({
        'file_path': file_paths,
        'emotion': labels,
        'encoded_label': encoded_labels
    })
    metadata.to_csv('../data/metadata.csv', index=False)
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()