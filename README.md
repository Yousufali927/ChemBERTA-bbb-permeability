<div align="center">

# DD 2.7: BBB Permeability Predictor

Fine-tuned **ChemBERTa-77M-MLM** for binary blood-brain barrier (BBB) permeability prediction on the BBBP dataset.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Model](https://img.shields.io/badge/Model-Hugging%20Face-orange)](https://huggingface.co/Yousuf7/ChemBERT-BBB-Permeability)
[![Gradio Demo](https://img.shields.io/badge/Demo-Live-green)](https://huggingface.co/spaces/Yousuf7/chembert-bbb-demo)

</div>

## Quick Overview

- **Task**: Binary classification — can a small molecule cross the blood-brain barrier? (Permeable vs Non-permeable)
- **Uses**: Filtering Molecules before In-vivo(lab) Screening and saving cost
- **Base Model**: ChemBERTa-77M-MLM (pre-trained on PubChem SMILES)
- **Dataset**: BBBP (~2,050 molecules from MoleculeNet / Therapeutics Data Commons)
- **Training**: Manual PyTorch loop (no Hugging Face Trainer due to accelerate/MPS version compatibility issues on M1 MacBook Air)
- **Hardware**: MacBook Air M1 (2020, 8GB RAM), CPU-only
- **Training duration**: ~25–30 minutes for 13 epochs

## Results

### Random Split (less realistic, higher scores due to scaffold leakage)

| Metric     | Value   | Comment                             |
|------------|---------|-------------------------------------|
| Accuracy   | 91.41%  | Very high overall                   |
| F1 Score   | 94.57%  | Strong on majority (permeable) class |
| ROC-AUC    | 0.9017  | Excellent separation                |
| PR-AUC     | 0.9490  | Very good precision-recall          |

### Scaffold Split (more realistic benchmark — no scaffold leakage)

| Metric     | Value   | Comment                             |
|------------|---------|-------------------------------------|
| Accuracy   | 84.48%  | Solid real-world generalization     |
| F1 Score   | 90.38%  | Good balance                        |
| ROC-AUC    | 0.8321  | Strong — in top range of published BBBP results |
| PR-AUC     | 0.9413  | Excellent                           |

**Benchmark context**:  
On BBBP, typical random-split ROC-AUC is 0.85–0.92, scaffold-split is 0.80–0.88 (MoleculeNet, TDC papers, ChemProp, etc.). This models scaffold ROC-AUC is **0.8321**  — lightweight model .

## Live Interactive Demo

Try predictions instantly (no code needed):  
**[Gradio Demo → BBB Predictor](https://huggingface.co/spaces/Yousuf7/chembert-bbb-demo)**

Features:
- Enter any SMILES string
- Get Permeable / Non-Permeable prediction + probability
- See molecule visualization
- Pre-loaded examples (Ethanol, Aspirin, Caffeine, Doxorubicin)

## Inferencing Model

```python
# Install once (if needed)
# pip install transformers torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load directly from Hugging Face (downloads automatically)
model_id = "Yousuf7/ChemBERT-BBB-Permeability"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

print("Model loaded successfully!")
