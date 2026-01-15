# Toxic Comment Classification: Bag-of-Words Representation Analysis

A comprehensive machine learning project exploring how different Bag-of-Words representations affect neural network performance in toxic comment detection using the Jigsaw Toxic Comment Classification dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Focus](#project-focus)
- [Methodology](#methodology)
- [Text Preprocessing](#text-preprocessing)
- [Representation Comparisons](#representation-comparisons)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)

## Overview

This project tackles toxic comment classification with a unique focus: **systematically comparing different Bag-of-Words representations** to understand their impact on neural network training and performance.

Rather than optimizing for maximum accuracy, this project investigates:
- How Count, Binary, TF-IDF, and Frequency representations differ in practice
- The role of normalization in Sklearn vs Keras pipelines
- Whether preprocessing complexity always improves model performance

## Dataset

**Source:** Jigsaw Toxic Comment Classification Challenge

**Statistics:**
- Total comments: ~160,000
- Class distribution: 90.4% non-toxic, 9.6% toxic
- 6 toxicity categories (multi-label classification)
- Highly imbalanced dataset

**Category Distribution:**
- Toxic: 15,294 (9.6%)
- Obscene: 8,449 (5.3%)
- Insult: 7,877 (4.9%)
- Severe Toxic: 1,595 (1.0%)
- Identity Hate: 1,405 (0.9%)
- Threat: 478 (0.3%)

## Project Focus

### Core Research Questions
1. How do different BoW representations (Count, Binary, TF-IDF, Frequency) affect model performance?
2. What is the impact of automatic normalization in Sklearn vs manual implementation in Keras?
3. Does more aggressive text preprocessing always lead to better results?

### Key Experiments
- **8 representation configurations tested:**
  - Sklearn: Count, Binary, TF-IDF
  - Sklearn Count with manual normalization
  - Keras: Count, Binary, TF-IDF, Frequency

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of toxic vs non-toxic comments
- Category frequency visualization
- Co-occurrence matrix revealing label correlations
- UpSet plot for multi-label combination analysis
- Interactive sample exploration

### 2. Data Balancing
- Undersampling majority class to create 50/50 toxic/non-toxic split
- Stratified train-test split maintaining class proportions
- Balanced dataset for fair model evaluation

### 3. Controlled Representation Testing
Each representation was tested under identical conditions:
- Same neural network architecture
- Same training parameters
- Same preprocessing pipeline
- Same balanced dataset

## Text Preprocessing

Comprehensive 9-step preprocessing pipeline:

1. **Lowercase Conversion** - Text standardization
2. **Unicode Normalization** - ASCII conversion, emoji removal
3. **Word Repetition Removal** - Spam pattern handling
4. **Laughter Normalization** - Standardizing excessive laughter
5. **Character Repetition Normalization** - Limiting character repetition
6. **Contraction Expansion** - 15+ common contractions
7. **URL & Email Removal** - Noise reduction
8. **Non-alphabetic Character Removal** - Keeping only letters
9. **Stopword Removal & Lemmatization** - Semantic normalization

**For full preprocessing details, see:** [Text Preprocessing Pipeline Documentation](https://github.com/soliman-benkhalil/Toxic-Comment-Classification)

## Representation Comparisons

### Sklearn Representations
- **Automatic L2 normalization** applied to all vectors
- CountVectorizer, TfidfVectorizer with `max_features=5000`
- Binary mode available through `binary=True` parameter

### Keras Representations
- **No automatic normalization**
- Manual tokenization and vectorization
- Four modes: Count, Binary, TF-IDF, Frequency
- Greater control over representation details

### Manual Normalization Test
- Implemented L1 normalization on Sklearn Count representation or just  X = X / X.sum(axis=1)
- Tested whether explicit normalization improves performance
- **Result:** Surprisingly degraded performance (87.76% → 85.82%)

## Model Architecture
```
Input Layer (5000 features)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Callbacks: Early Stopping (patience=5), ReduceLROnPlateau

## Results

### 🏆 Best Performance
**Configuration:** Keras TF-IDF  
**Test Accuracy:** 88.98%  
**Test Loss:** 0.2635  
**Convergence:** 7 epochs  

### Complete Results Comparison

| Representation | Test Accuracy | Test Loss | Best Val Accuracy | Epochs |
|---|---|---|---|---|
| **keras_tfidf** | **0.8898** | **0.2635** | **0.8957** | **7** |
| keras_count | 0.8868 | 0.3033 | 0.8888 | 10 |
| keras_freq | 0.8821 | 0.4081 | 0.8861 | 14 |
| sklearn_count | 0.8776 | 0.2969 | 0.8880 | 9 |
| keras_binary | 0.8760 | 0.2902 | 0.8826 | 9 |
| sklearn_binary | 0.8753 | 0.2949 | 0.8805 | 9 |
| sklearn_tfidf | 0.8737 | 0.2941 | 0.8809 | 10 |
| sklearn_count + manual norm | 0.8582 | 0.3757 | 0.8632 | 12 |

### Key Insights

1. **TF-IDF representations outperformed count-based methods** across both frameworks
2. **Keras implementations slightly outperformed Sklearn** despite lacking automatic normalization
3. **Manual normalization degraded performance**, suggesting neural networks may adapt better to non-normalized inputs in this context
4. **Representation choice impacts accuracy by ~3%** across identical architectures
5. **Faster convergence with TF-IDF** - best model converged in just 7 epochs

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow upsetplot
```

### NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Usage

### 1. Load Data
```python
train_df = pd.read_csv('train.csv')
```

### 2. Preprocess
```python
train_df['clean_text'] = train_df['comment_text'].apply(clean_text)
```

### 3. Balance Dataset
```python
balanced_df = create_balanced_dataset(train_df)
```

### 4. Test Different Representations
```python
# Sklearn TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_bow = vectorizer.fit_transform(balanced_df['clean_text']).toarray()

# Keras TF-IDF
X_bow = create_keras_tfidf(balanced_df['clean_text'], max_features=5000)
```

### 5. Train and Evaluate
```python
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[...])
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

## Future Work

### Planned Enhancements

1. **Preprocessing Impact Analysis**
   - Quantify effect of preprocessing intensity on each representation
   - Compare raw vs heavily cleaned text under identical conditions

2. **Full Dataset Training**
   - Scale experiments to complete 160K dataset
   - Leverage stronger computational resources

3. **Multi-Label Classification**
   - Predict all 6 toxicity categories simultaneously
   - Compare binary relevance vs chain classifiers

4. **Advanced Architectures**
   - Word2Vec/GloVe embeddings
   - LSTM/GRU sequence models
   - Transformer-based approaches (BERT, RoBERTa)

5. **Ensemble Methods**
   - Combine best-performing BoW representations
   - Stack with embedding-based models

---

## 📧 Contact

Questions or collaboration opportunities? Feel free to reach out!

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional representation experiments
- Alternative preprocessing techniques
- Cross-validation implementation
- Detailed error analysis
