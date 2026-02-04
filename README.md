# InkFlowBangla: Bengali Handwriting Text Generation

## 📢 Introduction
**InkFlowBangla** is a state-of-the-art few-shot diffusion model tailored for generating stylized Bengali handwritten text. Built upon the foundation of [DiffusionPen](https://github.com/koninik/DiffusionPen), this project introduces key enhancements specifically designed for the complexities of the Bengali script.

### Key Features
- **Grapheme Tokenizer**: Unlike standard tokenizers, InkFlowBangla employs a specialized grapheme-based tokenizer to accurately capture the structural components of Bengali characters.
- **Matra-aware Loss**: A novel loss function designed to preserve the "Matra" (the characteristic horizontal line in Bengali script), ensuring higher structural fidelity and consistency in generated text.
- **Few-shot Style Adaptation**: Capable of learning and imitating a writer's unique style from as few as five reference samples.

## 🚀 Getting Started

### Prerequisites
Ensure you have the necessary dependencies installed. The project relies on PyTorch, Diffusers, and Transformers.

### Project Structure
The codebase is organized as a modular Python package:
```text
codr/
├── src/
│   ├── training/    # Training scripts (e.g., train_grapheme_with_matra.py)
│   ├── inference/   # Inference scripts (e.g., infer_with_grapheme.py)
│   ├── models/      # UNet, Tokenizer, Feature Extractor
│   └── utils/       # Helper functions
```

## 💻 Usage

Running scripts as modules from the project root is recommended.

### Training

#### Grapheme-based Training (Recommended)
To train the model using the Grapheme Tokenizer and Matra-aware loss:
```bash
python -m src.training.train_grapheme_with_matra --pickle_path ./path/to/bengali_dataset.pickle --save_path ./checkpoints/inkflow_grapheme
```

#### Word-level Training
To train the standard word-level model (adapted from DiffusionPen):
```bash
python -m src.training.bn_train --pickle_path ./path/to/bengali_dataset.pickle
```

### Inference

Generate stylized Bengali text using the trained model:
```bash
python -m src.inference.infer_with_grapheme --model_path ./checkpoints/model.pth --text "আমার সোনার বাংলা" --mode sentence
```

**Common Inference Arguments:**
- `--text`: The Bengali text to generate.
- `--mode`: Generation mode (`word`, `sentence`, `paragraph`).
- `--style_image_paths`: Paths to reference handwriting images (for style cloning).

## 📄 Citation
InkFlowBangla is based on **DiffusionPen**. If you use this code, please cite the original paper:

```bibtex
@article{nikolaidou2024diffusionpen,
  title={DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author={Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2409.06065},
  year={2024}
}
```
