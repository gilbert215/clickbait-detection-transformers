# Clickbait Detection with Transformer Models

## What This Project Does
This project implements a binary classification system to automatically detect clickbait headlines using transformer-based neural networks. The system distinguishes between legitimate news headlines (non-clickbait) and misleading sensationalized headlines (clickbait) using fine-tuned pre-trained language models.

**Applications:**
- Content moderation for news platforms  
- Media literacy tools  
- Social media feed quality control  
- Information verification systems  

## Project Structure
- **Models**: BERT and ModernBERT with MLP classification head  
- **Dataset**: `christinacdl/clickbait_notclickbait_dataset` (binary labels: clickbait vs. non-clickbait)  
- **Training Pipeline**:  
  - Preprocessing (tokenization, padding, batching)  
  - Model training (cross-entropy loss, Adam/AdamW optimizers, learning rate tuning)  
  - Hyperparameter optimization (grid search, validation-based selection)  
  - Fine-tuning strategies (frozen base vs. full fine-tuning)  
- **Evaluation**: Accuracy, confusion matrix, error analysis, confidence distribution  

## Results

### Overall Performance
| Metric                | Value   |
|------------------------|---------|
| Total Examples         | 2191    |
| Correct Predictions    | 1996    |
| Incorrect Predictions  | 195     |
| Accuracy               | 91.12%  |
| Error Rate             | 8.90%   |
| False Positive Rate    | 4.14%   |
| False Negative Rate    | 16.97%  |

### Confusion Matrix
- **True Positives (TP)**: 675  
- **True Negatives (TN)**: 1321  
- **False Positives (FP)**: 57  
- **False Negatives (FN)**: 138  

### Confidence Analysis
- Correct predictions confidence: **0.933 ± 0.101**  
- Incorrect predictions confidence: **0.738 ± 0.151**  
- High-confidence errors (>80%): **74 (37.9% of errors)**  

### Key Insights
1. The model struggles more with **false negatives** (missed clickbait).  
2. Errors often occur with **high confidence**, suggesting overconfidence in borderline cases.  
3. Headlines with **numbers, quotes, or rhetorical phrasing** are prone to misclassification.  
