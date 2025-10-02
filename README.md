# Fine-Tuning Llama 3.2 3B on Particle Physics Q&As

This project fine-tunes a small language model (Llama 3.2 3B) on question-answer pairs generated from particle physics research papers, creating a specialized model for answering questions about experimental particle physics, detector technology, and data analysis methods.

## Project Overview

**Goal**: Create a specialized LLM that can answer technical questions about particle physics experiments, focusing on:
- D0 detector components at the Tevatron
- Neural networks for particle identification
- Calorimetry and jet physics
- Experimental data analysis methods

**Base Model**: `unsloth/Llama-3.2-3B-Instruct`  
**Fine-tuning Method**: QLoRA (4-bit quantization with Low-Rank Adaptation)  
**Training Platform**: Google Colab (Free tier with T4 GPU)  
**Final Model**: [bandurin/physics-llm-3b-finetuned](https://huggingface.co/bandurin/physics-llm-3b-finetuned)

## Project Structure

```
├── FineTune_Llama3_using_Physics_QnAs.ipynb  # Training notebook
├── test_physics_model.ipynb                   # Testing notebook
├── QnA.json                                   # Training dataset (50 Q&A pairs)
└── README.md                                  # This file
```

## Dataset

- **Source**: 25 physics research papers (particle physics domain)
- **Format**: 50 question-answer pairs in JSON format
- **Split**: 90% training (45 samples) / 10% validation (5 samples)
- **Topics Covered**:
  - D0 detector architecture
  - Neural network training algorithms 
  - Electromagnetic and hadronic calorimeters
  - Jet physics and particle identification
  - Higgs boson searches (diphoton channels)

## Training Configuration

### Model Architecture
- **Base Model**: Llama 3.2 3B Instruct
- **Quantization**: 4-bit (QLoRA)
- **LoRA Settings**:
  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Training Hyperparameters
- **Epochs**: 3
- **Batch Size**: 2 per device
- **Gradient Accumulation Steps**: 4 (effective batch size: 8)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **LR Scheduler**: Cosine
- **Weight Decay**: 0.01
- **Max Sequence Length**: 2048 tokens
- **Warmup Steps**: 5
- **Max Gradient Norm**: 0.3

### Hardware & Performance
- **GPU**: NVIDIA T4 (Google Colab Free Tier)
- **Training Time**: ~30-60 minutes
- **Memory Optimization**: Unsloth framework (2-5x speedup)
- **Precision**: FP16 (or BF16 if supported)

## System Prompt

The model was trained with the following system prompt:

```
You are an expert in particle physics, specializing in experimental 
techniques at collider experiments like the Tevatron and LHC. You have 
deep knowledge of neural networks for particle identification, jet physics, 
calorimetry, and data analysis methods. Provide accurate, detailed responses 
citing experimental methods and results when relevant.
```

## Setup & Installation

### Training (Google Colab)

```bash
# Install dependencies
!pip install --upgrade pyarrow>=14.0.0
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0"
!pip install -q pypdf2 pandas datasets

# Upload QnA.json to Colab
# Run FineTune_Llama3_using_Physics_QnAs.ipynb
```

### Testing (Local or Colab)

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install torch transformers

# Run test_physics_model.ipynb
```

## Usage

### Loading the Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="bandurin/physics-llm-3b-finetuned",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
```

### Inference

```python
SYSTEM_PROMPT = """You are an expert in particle physics..."""

def ask_question(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant\n\n")[-1]

# Test
answer = ask_question("What are the main components of the D0 detector?")
print(answer)
```

## Example Questions

The model can answer questions like:

1. "What are the main components of the D0 detector at the Tevatron?"
2. "How do electromagnetic calorimeters work in particle detectors?"
3. "What is the Manhattan algorithm in neural network training?"
4. "Explain jet physics in particle colliders."
5. "How are neural networks used for particle identification?"

## Training Results

- **Training Loss**: Successfully converged over 3 epochs
- **Model Size**: ~2.3 GB (merged model) or ~50-100 MB (LoRA adapters only)
- **Trainable Parameters**: 24.3M / 3.2B total (0.75%)

## Key Technologies

- **Unsloth**: Fast fine-tuning framework with optimizations
- **QLoRA**: Memory-efficient training with 4-bit quantization
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning
- **TRL (Transformer Reinforcement Learning)**: SFTTrainer for supervised fine-tuning
- **HuggingFace**: Model hosting and distribution

## Future Improvements

1. **Expand Dataset**: Generate 500-1000 Q&A pairs from additional physics papers
2. **Evaluate Performance**: Systematic evaluation on held-out physics questions
3. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, epochs
4. **Domain Expansion**: Include more physics domains (cosmology, quantum field theory)
5. **Deployment**: Deploy with vLLM or Ollama for production inference
6. **RAG Integration**: Combine with retrieval-augmented generation for paper citations

## Notes

- The model is specialized for particle physics and may not perform well on general questions
- Training used synthetic Q&As generated from papers, not human-annotated data
- Small dataset (50 samples) limits generalization; more data recommended for production use


## Citation

If you use this model, please cite:

```bibtex
@misc{physics-llm-finetuning,
  author = {Your Name},
  title = {Fine-Tuning Llama 3.2 3B on Particle Physics Q&As},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/bandurin/physics-llm-3b-finetuned}}
}
```

## License

This project uses Llama 3.2, which is subject to Meta's Llama 3 Community License Agreement.

## Acknowledgments

- **Unsloth AI**: For the efficient fine-tuning framework
- **Meta AI**: For the Llama 3.2 base model
- **HuggingFace**: For model hosting and transformers library
- **Google Colab**: For providing free GPU access

---

**Author**: bandurin  
**HuggingFace**: [https://huggingface.co/bandurin](https://huggingface.co/bandurin)  
**Date**: October 2025