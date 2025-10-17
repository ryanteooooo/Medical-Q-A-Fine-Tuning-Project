# 🏥 Medical Q&A Fine-Tuning Project

## What is this Project?

This project demonstrates fine-tuning a language model (DistilGPT-2) for medical question-answering tasks. It uses the MedQuAD dataset to train a model that can provide accurate medical answers. The project includes training, evaluation, and an interactive testing script.

## 📁 Project Tree

```
Medical_QA_Simplified/
├── Medical_QA_Training_Simplified.ipynb  ← Main training notebook
├── README.md                             ← This documentation
├── run_config.json                       ← Training configuration (hyperparameters)
├── test_model.py                          ← Interactive testing script
├── final_model/                           ← Saved fine-tuned model
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
└── training_history/                      ← Training results and summaries
    ├── results_run1.json
    ├── results_run2.json
    ├── ...
    ├── results_run10.json
    ├── training_summary_run1.txt
    ├── training_summary_run2.txt
    ├── ...
    └── training_summary_run10.txt
```

## 🚀 Quick Start

1. **Open the notebook**: `Medical_QA_Training_Simplified.ipynb`
2. **Run all cells** from top to bottom
3. **Test the model**: Run `python test_model.py`

## 📝 Output Files

### Training History
- **`results_runX.json`**: JSON files containing training results, hyperparameters, and evaluation metrics for each run
- **`training_summary_runX.txt`**: Text summaries of training runs with key statistics and analysis
- **`training_curves.png`**: PNG image showing loss curves and training progress visualization

## 🧪 How to Run test_model.py

The `test_model.py` script provides an interactive interface to test your trained medical Q&A model.

### Prerequisites
- The model must be trained and saved in the `final_model/` folder
- Python 3.x with required packages (transformers, torch)

### Running the Script
```bash
cd Medical_QA_Simplified
python test_model.py
```

### Interactive Features
- Ask medical questions and get AI-generated answers
- Adjust temperature with `temp X` (0.1-1.5) for different response styles
- Type `quit` or `exit` to stop

### Example Usage
```
❓ Your Question: What causes diabetes?
🤖 Model Answer:
Diabetes is caused by...
```

## 🎯 Final Model Folder

The `final_model/` folder contains the saved fine-tuned model after training completion. This folder is used to:

- **Store the trained model**: Contains all necessary files to load and use the model
- **Enable inference**: The model can be loaded by `test_model.py` or other scripts
- **Deployment**: Files can be used to deploy the model in production environments

### Key Files in final_model/:
- `model.safetensors`: The trained model weights
- `tokenizer.json`: Tokenizer configuration
- `config.json`: Model configuration
- Other supporting files for complete model functionality

## 📚 References

- **Model**: [DistilGPT-2](https://huggingface.co/distilgpt2)
- **Dataset**: [MedQuAD](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- **MMLU**: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
