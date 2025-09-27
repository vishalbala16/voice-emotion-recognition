# Voice Emotion Recognition - AI/ML Internship Project

## Overview
Complete implementation of voice emotion recognition system using classical ML and deep learning approaches.

## Quick Start
```bash
# Install dependencies
pip install -r requirements_full.txt

# Run complete pipeline
python src/full_pipeline.py

# Launch web app
streamlit run streamlit_app/app_full.py
```

## Project Structure
```
voice_ai_project/
├── data/                   # Dataset and processed features
├── src/                    # Core implementation
├── models/                 # Trained models (6 models)
├── results/                # Evaluation results
├── streamlit_app/          # Web application
├── notebooks/              # Analysis notebook
└── docs/                   # Final report
```

## Features
- 8 emotion classification (RAVDESS-style dataset)
- Classical ML: Random Forest, Logistic Regression, SVM
- Deep Learning: CNN, RNN, Transformer
- Interactive web interface
- Comprehensive evaluation and visualization

## Results
- Classical ML: 100% accuracy
- CNN: 46.2% accuracy
- Complete feature extraction (MFCC, Mel-spectrogram, Chroma)
