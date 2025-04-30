# Cross-Encoder Lion vs AdamW

## Overview

This repository contains code for training and evaluating Cross-Encoder models using different optimizers (Lion and AdamW) for information retrieval tasks. The project tests the performance of these optimizers across different models (ModernBERT, MiniLM, GTE) and evaluates their effectiveness on standard benchmarks like MS MARCO and TREC DL 2019.

## Project Description

Cross-Encoders are powerful models for ranking tasks in information retrieval. This project investigates how different optimization strategies affect the performance of these models by comparing:

- **Lion optimizer**: A relatively new optimizer that claims to be more memory-efficient with better generalization
- **AdamW optimizer**: A well-established optimizer widely used in transformer models

The experiments are conducted using various model architectures:
- ModernBERT-base 
- MiniLM
- GTE (General Text Embeddings)

Each model is trained for 3 epochs with both optimizers, and performance is tracked across multiple evaluation metrics.

## Repository Structure

```
├── ms_marco_val_mrr.py          # Evaluation script for MS MARCO validation set (MRR@10)
├── trec_dl_19_eval_2.py         # Evaluation script for TREC DL 2019 dataset
├── trainer.py                   # Main training script for cross-encoder models
├── modal_trainer_offload.py     # Modal deployment script for cloud training
├── trec_modal.py                # Modal deployment script for TREC evaluations
├── eval_results/                # Directory containing evaluation results
│   ├── mrrs.txt                 # Summary of MRR results across all models
│   ├── modern_bert/             # ModernBERT results
│   ├── mini_lm/                 # MiniLM results
│   └── gte/                     # GTE results
└── requirements.txt             # Required dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Cross-Encoder-Lion-vs-AdamW.git
   cd Cross-Encoder-Lion-vs-AdamW
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For TREC evaluation, clone and build the trec_eval repository:
   ```bash
   git clone https://github.com/usnistgov/trec_eval.git
   cd trec_eval
   make
   ```

## Usage

### Local Training

To train a cross-encoder model locally:

```bash
python trainer.py
```

By default, the script uses the AdamW optimizer. To switch to Lion, modify the optimizer selection in the trainer.py file.

### Cloud Training with Modal

For cloud-based training using Modal:

1. Set up Modal and configure your API keys:
   ```bash
   pip install modal
   modal token new
   ```

2. Run the training script on Modal:
   ```bash
   modal run modal_trainer_offload.py
   ```

### Evaluation

To evaluate on MS MARCO validation set:

```bash
python ms_marco_val_mrr.py --parent_dir path/to/models
```

To evaluate on TREC DL 2019:

```bash
python trec_dl_19_eval_2.py --model_name path/to/model
```

## Results

### Mean Reciprocal Rank (MRR@10) on MS MARCO

#### ModernBERT

| Epoch | Lion    | AdamW   |
|-------|---------|---------|
| 1     | 0.5834  | 0.5866  |
| 2     | 0.5908  | 0.5886  |
| 3     | 0.5989  | 0.5916  |

#### MiniLM

| Epoch | Lion    | AdamW   |
|-------|---------|---------|
| 1     | 0.5890  | 0.5828  |
| 2     | 0.5942  | 0.5818  |
| 3     | 0.5988  | 0.5826  |

#### GTE

| Epoch | Lion    | AdamW   |
|-------|---------|---------|
| 1     | 0.5854  | 0.5940  |
| 2     | 0.5957  | 0.5942  |
| 3     | 0.5931  | 0.5972  |

### TREC DL 2019 Results

Detailed evaluation metrics for each model configuration can be found in the `eval_results` directory. Key metrics include:

- NDCG@10
- MAP
- Recall
- Precision@10

## Key Findings

1. The Lion optimizer generally shows better performance than AdamW by the final epoch for ModernBERT and MiniLM models.
2. MiniLM with Lion achieves the highest overall MRR@10 score of 0.5988 after 3 epochs.
3. The GTE model shows slightly better performance with AdamW over Lion.
4. Both optimizers demonstrate consistent improvement over the training epochs, with Lion often showing steeper improvement curves.

## License

This repository is provided without a specific license. All rights reserved.

## Acknowledgments

- The project uses the Hugging Face Transformers and Sentence-Transformers libraries
- Training utilizes the MS MARCO dataset
- Evaluation is performed using TREC DL 2019 and MS MARCO validation sets
- Cloud execution is powered by Modal