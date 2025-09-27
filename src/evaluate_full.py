import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

def evaluate_all_models():
    """Evaluate all trained models and create comprehensive comparison"""
    # Load data
    features = np.load('../data/classical_features.npy')
    spectrograms = np.load('../data/spectrograms.npy')
    labels = np.load('../data/labels.npy')
    label_encoder = joblib.load('../models/label_encoder.pkl')
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    results = {}
    
    # Evaluate Classical ML models
    classical_models = ['random_forest', 'logistic_regression', 'svm']
    for model_name in classical_models:
        model = joblib.load(f'../models/{model_name}_model.pkl')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[model_name.replace('_', ' ').title()] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'type': 'Classical ML'
        }
    
    # Evaluate Deep Learning models
    spectrograms_cnn = spectrograms[..., np.newaxis]
    _, X_test_cnn, _, y_test_dl = train_test_split(
        spectrograms_cnn, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # CNN
    cnn_model = tf.keras.models.load_model('../models/cnn_model.h5')
    cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    cnn_accuracy = accuracy_score(y_test_dl, cnn_pred)
    
    results['CNN'] = {
        'accuracy': cnn_accuracy,
        'y_test': y_test_dl,
        'y_pred': cnn_pred,
        'type': 'Deep Learning'
    }
    
    # RNN
    _, X_test_rnn, _, _ = train_test_split(
        spectrograms, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    rnn_model = tf.keras.models.load_model('../models/rnn_model.h5')
    rnn_pred = np.argmax(rnn_model.predict(X_test_rnn), axis=1)
    rnn_accuracy = accuracy_score(y_test_dl, rnn_pred)
    
    results['RNN'] = {
        'accuracy': rnn_accuracy,
        'y_test': y_test_dl,
        'y_pred': rnn_pred,
        'type': 'Deep Learning'
    }
    
    # Transformer
    transformer_model = tf.keras.models.load_model('../models/transformer_model.h5')
    transformer_pred = np.argmax(transformer_model.predict(X_test_rnn), axis=1)
    transformer_accuracy = accuracy_score(y_test_dl, transformer_pred)
    
    results['Transformer'] = {
        'accuracy': transformer_accuracy,
        'y_test': y_test_dl,
        'y_pred': transformer_pred,
        'type': 'Deep Learning'
    }
    
    return results, label_encoder.classes_

def create_comprehensive_visualizations(results, class_names):
    """Create all required visualizations"""
    import os
    os.makedirs('../results', exist_ok=True)
    
    # 1. Model Accuracy Comparison
    plt.figure(figsize=(12, 8))
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    colors = ['skyblue' if results[model]['type'] == 'Classical ML' else 'lightcoral' for model in models]
    
    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Classical ML vs Deep Learning Performance Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='Classical ML'),
                      Patch(facecolor='lightcoral', label='Deep Learning')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        if i < len(axes):
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature Importance (for Random Forest)
    rf_model = joblib.load('../models/random_forest_model.pkl')
    feature_names = [f'MFCC_{i}' for i in range(13)] + \
                   [f'Chroma_{i}' for i in range(12)] + \
                   ['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth', 'Zero_Crossing_Rate']
    
    plt.figure(figsize=(12, 8))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.bar(range(15), importances[indices])
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_reports(results, class_names):
    """Generate detailed classification reports"""
    reports = {}
    for name, result in results.items():
        report = classification_report(result['y_test'], result['y_pred'], 
                                     target_names=class_names, output_dict=True)
        reports[name] = report
        
        print(f"\n{name.upper()} CLASSIFICATION REPORT:")
        print("="*50)
        print(classification_report(result['y_test'], result['y_pred'], target_names=class_names))
    
    return reports

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Evaluate all models
    results, class_names = evaluate_all_models()
    
    # Create visualizations
    create_comprehensive_visualizations(results, class_names)
    
    # Generate classification reports
    reports = generate_classification_reports(results, class_names)
    
    # Save comprehensive results
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Type': result['type'],
            'Accuracy': result['accuracy'],
            'Precision': reports[name]['macro avg']['precision'],
            'Recall': reports[name]['macro avg']['recall'],
            'F1-Score': reports[name]['macro avg']['f1-score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('../results/comprehensive_evaluation.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("Results saved to results/ directory")
    print("="*70)

if __name__ == "__main__":
    main()