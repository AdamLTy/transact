# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the 1st place solution for the WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge by Team momo. The solution is based on FuxiCTR framework and implements custom neural network models for multimodal click-through rate prediction.

## Environment Setup

Create conda environment and install dependencies:
```bash
conda create -n fuxictr_momo python==3.9
pip install -r requirements.txt
source activate fuxictr_momo
```

Key dependencies: fuxictr==2.3.7, torch==1.13.1+cu117, numpy==1.26.4

## Common Commands

### Training and Prediction
- **Run complete pipeline**: `sh ./run.sh`
- **Train model**: `python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0`
- **Make predictions**: `python prediction.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0`
- **Run parameter tuning**: `python run_param_tuner.py`

### GPU Usage
Use `--gpu 0` for GPU training or `--gpu -1` for CPU-only mode.

## Architecture Overview

The codebase implements three main model architectures:

1. **DIN** (`src/DIN.py`) - Deep Interest Network with attention mechanism
2. **Transformer_DCN** (`src/Transformer_DCN.py`) - Main solution combining Transformer and DCNv2
3. **Transformer_DCN_Quant** (`src/Transformer_DCN_Quant.py`) - Quantized version with Vector/Residual Quantization

### Key Components

- **Models**: Located in `src/` directory, inherit from FuxiCTR's `BaseModel`
- **Data Loading**: Custom `MMCTRDataLoader` in `src/mmctr_dataloader.py`
- **Configuration**: YAML configs in `config/` directory with separate model and dataset configs
- **Execution Scripts**: `run_expid.py` for training, `prediction.py` for inference

### Configuration Structure

Each model has a main config file (e.g., `Transformer_DCN_microlens_mmctr_tuner_config_01.yaml`) that references:
- `model_config.yaml` - Model hyperparameters
- `dataset_config.yaml` - Data processing parameters

### Model Architecture Details

- **Transformer layers**: 2 layers with 256 dim_feedforward, 0.2 dropout
- **DCN cross layers**: 3 layers with [1024, 512, 256] hidden units
- **Embedding dimension**: 64
- **First k columns**: 16 (important architectural parameter)
- **Attention heads**: 1

## Data Pipeline

- Raw data should be placed in `data/` directory
- Uses FuxiCTR's feature processing pipeline
- Custom dataloader handles multimodal embeddings
- Supports both training and prediction modes

## Model Training Notes

- Best validation AUC achieved: 0.976603
- Test set AUC: 0.9839
- Uses accumulation steps for gradient accumulation
- Supports both CPU and GPU training
- Models are saved with experiment IDs for reproducibility