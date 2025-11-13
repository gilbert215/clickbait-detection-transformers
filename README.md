# Clickbait Detection with Transformer Models

## What This Project Does

This project implements a binary classification system to automatically detect clickbait headlines using transformer-based neural networks. Clickbait refers to sensationalized headlines designed to attract clicks at the expense of accuracy. The system distinguishes between legitimate news headlines (non-clickbait) and misleading sensationalized headlines (clickbait) using fine-tuned pre-trained language models.

**Applications:**
- Content moderation for news platforms
- Media literacy tools
- Social media feed quality control
- Information verification systems

## Models Description

### BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: 12-layer transformer encoder with 768 hidden dimensions
- **Parameters**: ~110M parameters
- **Tokenization**: WordPiece tokenizer with 30K vocabulary
- **Pre-training**: Trained on BooksCorpus and English Wikipedia

### ModernBERT
- **Architecture**: Updated transformer architecture with improved efficiency
- **Parameters**: ~139M parameters
- **Tokenization**: Enhanced tokenizer with better subword segmentation
- **Pre-training**: Trained on more recent and diverse text corpora

### Classification Head
Both models use a multi-layer perceptron (MLP) classifier on top of the transformer encoder:
- **Input**: [CLS] token representation from transformer (768-dim)
- **Hidden Layers**: 1-3 fully connected layers with configurable dimensions (64-256)
- **Activation**: ReLU
- **Output**: 2 classes (clickbait vs. non-clickbait)

**Architecture Options Explored:**
- Hidden layer sizes: 64, 128, 256
- Number of layers: 1, 2, 3
- Base model freezing: frozen vs. fine-tuned

## Methods

### Dataset
- **Source**: `christinacdl/clickbait_notclickbait_dataset` from Hugging Face
- **Labels**: Binary (0 = non-clickbait, 1 = clickbait)
- **Splits**: Train, Validation, Test

### Training Pipeline

1. **Data Preprocessing**
   - Tokenization using model-specific tokenizers
   - Padding and truncation to uniform length
   - Batch creation for efficient training

2. **Model Training**
   - Loss function: Cross-entropy loss
   - Optimizers tested: SGD, Adam, AdamW
   - Learning rates: 1e-3, 5e-4, 1e-4
   - Regularization: Weight decay (1e-4)
   - Batch sizes: 16, 32, 64
   - Training epochs: 1-5

3. **Hyperparameter Optimization**
   - Systematic grid search over hyperparameter space
   - Experiment tracking with Weights & Biases
   - Validation-based model selection

4. **Fine-tuning Strategies**
   - **Frozen base**: Only train classification head (faster, fewer parameters)
   - **Full fine-tuning**: Train entire model (better performance, more resources)

### Training Configuration
```python
Device: GPU (CUDA) / CPU / MPS (Apple Silicon)
Batch Processing: Collated batches with dynamic padding
Gradient Updates: Standard backpropagation
Early Stopping: Based on validation loss
Checkpoint Saving: Best model based on validation accuracy
```

## Metrics Used in Evaluation

### Primary Metrics

1. **Accuracy**
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```
   - Proportion of correctly classified headlines
   - Main metric for model comparison

2. **Loss (Cross-Entropy)**
   ```
   Loss = -Σ y_i * log(ŷ_i)
   ```
   - Measures prediction confidence
   - Used for training optimization

### Additional Metrics

3. **Confusion Matrix**
   - True Positives (TP): Correctly identified clickbait
   - True Negatives (TN): Correctly identified non-clickbait
   - False Positives (FP): Non-clickbait incorrectly labeled as clickbait
   - False Negatives (FN): Clickbait incorrectly labeled as non-clickbait

4. **Per-Class Performance**
   - Class-specific accuracy
   - Error analysis by category

### Evaluation Procedure
- **Validation Set**: Used during training for hyperparameter selection
- **Test Set**: Final evaluation on held-out data
- **Prediction Output**: Binary labels (0/1) for unlabeled test data



*Results will be updated upon completion of hyperparameter tuning experiments*
