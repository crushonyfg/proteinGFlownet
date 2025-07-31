# ICLR-GFlownet: Protein Design with Generative Flow Networks

This repository contains the implementation for our ICLR submission on protein design using Generative Flow Networks (GFlowNet). The project integrates state-of-the-art protein language models with reinforcement learning techniques to enable efficient protein sequence generation and optimization.

## Overview

Our approach combines ESM-IF model fine-tuning with GFlowNet-based generation to create a robust framework for protein design. The pipeline includes supervised fine-tuning (SFT), direct preference optimization (DPO), and GFlowNet implementation for exploring the protein sequence space efficiently.

## Repository Structure

```
├── datasets/           # Dataset preparation and storage
├── protein-dpo/        # ESM-IF model SFT and DPO training
├── gflownet/          # GFlowNet implementation
├── evaluation/        # Model performance evaluation code
├── models/            # Trained model storage
└── README.md          # This file
```

### Directory Details

#### `datasets/`
Contains scripts and utilities for:
- Dataset preparation and preprocessing
- Data storage and management
- Format conversion utilities
- Train/validation/test splits

#### `protein-dpo/`
Implementation of protein language model training based on the [ProteinDPO repository](https://github.com/evo-design/protein-dpo):
- **SFT (Supervised Fine-Tuning)**: ✅ Completed
- **DPO (Direct Preference Optimization)**: 🚧 In Progress
- ESM-IF model integration
- Training scripts and configurations

#### `gflownet/`
GFlowNet implementation for protein sequence generation:
- Based on concepts from [ProtRL](https://github.com/AI4PDLab/ProtRL)
- Custom reward functions for protein properties
- Efficient exploration of sequence space
- Integration with protein language models

#### `evaluation/`
Comprehensive evaluation framework for:
- Model performance assessment
- Protein property prediction accuracy
- Sequence quality metrics
- Comparative analysis tools

## Installation

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended)
- Conda or virtualenv

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd internTA

# Create and activate conda environment
conda create -n iclr-gflownet python=3.10
conda activate iclr-gflownet

# Install dependencies
pip install -r requirements.txt

# Install ESM model dependencies
pip install fair-esm
pip install biotite
```

## Quick Start

### 1. Dataset Preparation
```bash
cd datasets/
python prepare_dataset.py --input_file <your_protein_data> --output_dir ./processed
```

### 2. Protein-DPO Training
```bash
cd protein-dpo/
# SFT training (completed)
python sft_esmif.py --config configs/sft_config.yaml

# DPO training (in progress)
python dpo_training.py --config configs/dpo_config.yaml
```

### 3. GFlowNet Training
```bash
cd gflownet/
python train_gflownet.py --model_path ../protein-dpo/models/sft_model --config configs/gflownet_config.yaml
```

### 4. Evaluation
```bash
cd evaluation/
python evaluate_models.py --model_dir ../models --test_data ../datasets/test.csv
```

## Project Timeline

| Phase | Task | Deadline | Status |
|-------|------|----------|--------|
| Phase 1 | ProteinDPO Implementation | April 8 | ✅ SFT Complete, 🚧 DPO In Progress |
| Phase 2 | GFlowNet Implementation | August 11 | 📅 Planned |
| Phase 3 | Models & Evaluation | August 18 | 📅 Planned |
| Phase 4 | Extended Experiments | September 1 | 📅 Planned |
| Phase 5 | Final Review & Optimization | TBD | 📅 Planned |

## Key Features

- **Multi-Modal Training**: Combines SFT and DPO for robust protein language model training
- **Efficient Exploration**: GFlowNet-based sequence generation for better exploration
- **Comprehensive Evaluation**: Multiple metrics for assessing protein design quality
- **Modular Design**: Easy to extend and modify for different protein design tasks
- **GPU Optimized**: Efficient implementation for CUDA-enabled hardware

## Usage Examples

### Training a Custom Protein Model
```python
from protein_dpo import ESMIFTrainer
from gflownet import ProteinGFlowNet

# Initialize trainer
trainer = ESMIFTrainer(config='configs/custom_config.yaml')

# Train SFT model
sft_model = trainer.train_sft(dataset_path='datasets/protein_sequences.csv')

# Initialize GFlowNet
gfn = ProteinGFlowNet(base_model=sft_model)
gfn.train(reward_function=custom_reward_fn)
```

### Generating Novel Protein Sequences
```python
from gflownet import ProteinGenerator

generator = ProteinGenerator.load('models/trained_gflownet.pt')
sequences = generator.sample(n_samples=100, max_length=200)
```

## Evaluation Metrics

Our evaluation framework includes:
- **Protein Validity**: Structural and chemical feasibility
- **Diversity**: Sequence space coverage
- **Novelty**: Distance from training data
- **Property Optimization**: Target-specific metrics
- **Perplexity**: Language model confidence


## Dependencies

Key dependencies include:
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- ESM (fair-esm)
- NumPy, Pandas, Matplotlib
- Biotite for protein structure handling
- Weights & Biases for experiment tracking

## Citation

If you use this code in your research, please cite:

```bibtex
@article{iclr_gflownet_2024,
  title={Protein Design with Generative Flow Networks},
  author={[Your Name]},
  journal={ICLR},
  year={2024}
}
```

## Acknowledgments

- [ProtRL](https://github.com/AI4PDLab/ProtRL) for reinforcement learning framework inspiration
- [ProteinDPO](https://github.com/AI4PDLab/ProteinDPO) for the base protein language model training
- ESM team for the foundational protein language models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue or contact [your-email@domain.com].

---

**Status**: 🚧 Active Development | **Last Updated**: [Current Date] 