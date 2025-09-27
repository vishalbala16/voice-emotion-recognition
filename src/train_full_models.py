import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
import joblib
import pandas as pd

def train_classical_models():
    """Train classical ML models"""
    print("Training Classical ML Models...")
    
    # Load data
    features = np.load('../data/classical_features.npy')
    labels = np.load('../data/labels.npy')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save model
        model_filename = name.lower().replace(' ', '_') + '_model.pkl'
        joblib.dump(model, f'../models/{model_filename}')
        
        print(f"{name} accuracy: {accuracy:.4f}")
    
    return results

def build_cnn_model(input_shape, num_classes):
    """Build CNN model for mel-spectrogram classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_rnn_model(input_shape, num_classes):
    """Build RNN model for sequence classification"""
    model = Sequential([
        Reshape((input_shape[0], input_shape[1]), input_shape=input_shape),
        LSTM(64, return_sequences=True, dropout=0.3),
        LSTM(32, dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transformer_model(input_shape, num_classes):
    """Build simple Transformer-like model"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Reshape for 1D convolution
    x = tf.keras.layers.Reshape((input_shape[0] * input_shape[1], 1))(inputs)
    
    # 1D Convolution layers
    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_deep_learning_models():
    """Train deep learning models"""
    print("Training Deep Learning Models...")
    
    # Load data
    spectrograms = np.load('../data/spectrograms.npy')
    labels = np.load('../data/labels.npy')
    
    # Add channel dimension for CNN
    spectrograms_cnn = spectrograms[..., np.newaxis]
    
    # Split data
    X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(
        spectrograms_cnn, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_rnn, X_test_rnn = spectrograms[train_test_split(
        range(len(spectrograms)), test_size=0.2, random_state=42, stratify=labels
    )[0]], spectrograms[train_test_split(
        range(len(spectrograms)), test_size=0.2, random_state=42, stratify=labels
    )[1]]
    
    num_classes = len(np.unique(labels))
    results = {}
    
    # Train CNN
    print("Training CNN...")
    cnn_model = build_cnn_model(X_train_cnn.shape[1:], num_classes)
    
    history_cnn = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    cnn_model.save('../models/cnn_model.h5')
    cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)[1]
    
    results['CNN'] = {
        'model': cnn_model,
        'accuracy': cnn_accuracy,
        'history': history_cnn
    }
    
    # Train RNN
    print("Training RNN...")
    rnn_model = build_rnn_model(X_train_rnn.shape[1:], num_classes)
    
    history_rnn = rnn_model.fit(
        X_train_rnn, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    rnn_model.save('../models/rnn_model.h5')
    rnn_accuracy = rnn_model.evaluate(X_test_rnn, y_test, verbose=0)[1]
    
    results['RNN'] = {
        'model': rnn_model,
        'accuracy': rnn_accuracy,
        'history': history_rnn
    }
    
    # Train Transformer-like model
    print("Training Transformer...")
    transformer_model = build_transformer_model(X_train_rnn.shape[1:], num_classes)
    
    history_transformer = transformer_model.fit(
        X_train_rnn, y_train,
        epochs=25,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    transformer_model.save('../models/transformer_model.h5')
    transformer_accuracy = transformer_model.evaluate(X_test_rnn, y_test, verbose=0)[1]
    
    results['Transformer'] = {
        'model': transformer_model,
        'accuracy': transformer_accuracy,
        'history': history_transformer
    }
    
    print(f"CNN accuracy: {cnn_accuracy:.4f}")
    print(f"RNN accuracy: {rnn_accuracy:.4f}")
    print(f"Transformer accuracy: {transformer_accuracy:.4f}")
    
    return results

def main():
    print("="*60)
    print("TRAINING ALL MODELS - CLASSICAL ML + DEEP LEARNING")
    print("="*60)
    
    # Train classical models
    classical_results = train_classical_models()
    
    # Train deep learning models
    dl_results = train_deep_learning_models()
    
    # Save results summary
    all_results = []
    
    for name, result in classical_results.items():
        all_results.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Type': 'Classical ML'
        })
    
    for name, result in dl_results.items():
        all_results.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Type': 'Deep Learning'
        })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('../results/all_model_results.csv', index=False)
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\nModels saved in models/ directory")
    print("Results saved to results/all_model_results.csv")

if __name__ == "__main__":
    main()