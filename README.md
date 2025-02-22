# Natural Language Inference (NLI) with XLM-RoBERTa and LLM Inference

This repository contains the implementation of a Natural Language Inference (NLI) model using **XLM-RoBERTa** for classifying relationships between sentence pairs (Premise and Hypothesis) into three categories: **Entailment**, **Contradiction**, and **Neutral**. The project leverages the **SNLI (Stanford Natural Language Inference)** dataset and includes training, validation, and evaluation pipelines. Additionally, it supports **LLM-based inference** using models like GPT-4 and LLaMA for zero-shot NLI tasks.

---

## Key Features
- **Dataset**: SNLI dataset with 549,367 training, 9,842 validation, and 10,000 test examples.
- **Model**: Fine-tuned **XLM-RoBERTa** for sequence classification.
- **Preprocessing**:
  - Tokenization using XLM-RoBERTa tokenizer.
  - Dynamic padding and batching for efficient training.
- **Training**:
  - Cross-entropy loss and Adam optimizer.
  - Batch size of 64 and learning rate of 1e-5.
  - WandB integration for tracking training and validation metrics.
- **Evaluation**:
  - Validation accuracy and loss computed after each epoch.
  - Support for **GPT-4** and **LLaMA**-based inference for zero-shot NLI tasks.
- **LLM Inference**:
  - Zero-shot NLI using GPT-4 and LLaMA.
  - Example generation and prompt engineering for LLM-based classification.

---

## Results
- Achieved **high validation accuracy** (exact metrics logged in WandB).
- Demonstrated the effectiveness of XLM-RoBERTa for NLI tasks.
- Successfully performed zero-shot NLI using GPT-4 and LLaMA.

## Example Generation for LLM Inference
- Use the `get_examples` function to generate formatted sentence pairs for LLM-based inference:
  ```python
  from datasets import load_dataset

  def get_examples(dataset_name, val_size):
      dataset = load_dataset(dataset_name)
      valid_data = dataset['validation'].filter(lambda example: example['label'] != -1)
      val_examples = valid_data[:val_size]
      pairs = list(zip(val_examples['premise'], val_examples['hypothesis']))
      formatted_strings = [' </s> '.join(pair) for pair in pairs]
      result = '\n'.join(formatted_strings)
      return result, val_examples['label']
  ```

## Future Work
- Extend the model to support multilingual NLI tasks.
- Experiment with larger transformer models (e.g., GPT-4, T5).
- Incorporate additional datasets (e.g., MultiNLI) for improved generalization.
- Enhance LLM-based inference with advanced prompt engineering.
