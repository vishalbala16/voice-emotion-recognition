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

## Sample Outputs
<img width="1902" height="788" alt="out 1" src="https://github.com/user-attachments/assets/249329af-1b74-441f-8d1b-4cf72cd0274f" />
<img width="1892" height="565" alt="out 2" src="https://github.com/user-attachments/assets/1a6be32b-e99d-4c6e-8ce2-3506436ea857" />
<img width="1887" height="571" alt="out 3" src="https://github.com/user-attachments/assets/90404f79-453d-4000-9629-92e8c34f9d0c" />
<img width="403" height="671" alt="out 4" src="https://github.com/user-attachments/assets/2ecf1c8c-66ac-4536-8316-d5985776b1df" />
<img width="1882" height="882" alt="out 5" src="https://github.com/user-attachments/assets/ad537aac-a3ac-4ae9-ab1c-8b78ea8e7c5c" />
<img width="1460" height="899" alt="out 6" src="https://github.com/user-attachments/assets/4cc8e07f-8cc4-4b48-89af-242a468b54ea" />
<img width="1460" height="978" alt="out 7" src="https://github.com/user-attachments/assets/b9d6d989-7c76-445f-9595-34a634e8e631" />
<img width="1460" height="966" alt="out 8" src="https://github.com/user-attachments/assets/8a4dda19-b6e1-4500-861e-617663d8c34f" />







