# BioFoldNet
Leveraging Bidirectional LSTM and GRU for Advanced Protein Structure Prediction

## Overview  
BioFoldNet is a deep learning framework designed to enhance **protein fold recognition** using **Bidirectional LSTM (BiLSTM) and Bidirectional GRU (BiGRU)**. By leveraging the sequential dependencies of amino acid sequences, this project aims to improve **protein structure prediction**, a crucial task in **bioinformatics and drug discovery**.  

## Features  
- Utilizes **BiLSTM** and **BiGRU** architectures for sequence-based protein fold recognition.  
- Employs **ProtBERT-based tokenization** to extract meaningful sequence embeddings.  
- Benchmarked against state-of-the-art models like **ASFold-DNN** and **Conv-SXGbg-DeepFold**.  
- Achieves **90% accuracy with BiGRU**, outperforming traditional methods.  
- Trained on the **ASTRAL SCOPe 2.08** dataset with hierarchical protein classification.  
- Future scope includes **drug interaction prediction** and **explainable AI (XAI)** techniques.  

## Dataset  
- **Source**: [SCOPe 2.08 (2021)](http://scop.berkeley.edu/)  
- **Protein Classification Levels**:  
  - **Class**: Structural characteristics (alpha, beta proteins, etc.).  
  - **Fold**: Major structural similarities.  
  - **Superfamily**: Evolutionary relationships.  
  - **Family**: Sequence similarity and shared function.  
- **Train-Test Split**:  
  - **Training Samples**: 4,954  
  - **Testing Samples**: 1,240
 
**Data Preprocessing**
-Filters low-representation folds to ensure robust training.
-Uses BERT-based tokenization to convert amino acid sequences into embeddings.

**Model Training**
-BiLSTM and BiGRU architectures trained with:
-Hidden size: 128
-Dropout: 0.5
-Learning rate: 0.001
-Loss function: Cross-entropy
-Optimizer: Adam

## 🛠Installation  
### Dependencies  
Ensure you have the following installed:  
- Python 3.12  
- TensorFlow / PyTorch  
- Hugging Face Transformers  
- NumPy, Pandas, Matplotlib  
- Scikit-learn  

### Setup  
Clone the repository and install dependencies:  

