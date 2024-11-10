# Adversarial Alignment for Code Generation

This repository contains various approaches to improve code generation capabilities of language models through different training methods, including Direct Preference Optimization (DPO), Proximal Policy Optimization (PPO), and traditional Supervised Fine-Tuning (SFT).

## Overview

The project consists of three main components:
1. **DPO Training**: Aligns a student model's output distribution with a teacher model using preference learning
2. **PPO Training**: Reinforcement learning approach using code execution feedback
3. **SFT Training**: Traditional supervised fine-tuning as a baseline

## Project Structure

```
.
├── dpo/
│   ├── generate_dataset.py   # Generate preference datasets
│   └── train.py             # DPO training script
├── ppo/
│   ├── bob/                 # PPO implementation
│   └── ppo_train.py         # PPO training script
├── sft/                     # Supervised fine-tuning
└── microsoft_phi/           # Initial benchmarking
```

## Installation

```bash
# Build Docker image
docker build -t adversarial .

# Run with volume mount for results
docker run -v $(pwd)/results:/usr/src/app/generated_outputs adversarial
```

## DPO Training

The DPO approach uses two models (teacher and student) to generate a preference dataset and then trains the student model to align with the teacher's preferences.

### Generating Dataset

```bash
python dpo/generate_dataset.py \
    --oracle_path="path/to/teacher/model" \
    --student_path="path/to/student/model" \
    --num_subtopic_per_topic=10 \
    --num_exercise_per_subtopic=5 \
    --dataset_path="output/dataset/path"
```

### Training

```bash
python dpo/train.py \
    --oracle_path="path/to/teacher/model" \
    --student_path="path/to/student/model" \
    --output_dir="output/model/path" \
    --num_steps=200 \
    --learning_rate=5e-6 \
    --beta=0.01
```

## PPO Training

PPO training uses reinforcement learning where the reward is based on code execution metrics:
- Unit test performance
- Assertion errors
- Runtime errors
- Syntax errors

```bash
python ppo/ppo_train.py
```

## SFT Training

The SFT directory contains baseline supervised fine-tuning scripts for comparison with DPO and PPO approaches.

## Initial Benchmarking

The repository includes initial benchmarking of the Microsoft-phi-1.5 model on HumanEval:

```bash
# Run benchmark
docker build -t benchmark .
docker run -v $(pwd)/results:/usr/src/app/generated_outputs benchmark
```

### LoRA Parameters Analysis

The project includes LoRA parameter analysis for different models:
- StarCoderBase-1B: 835,584 trainable params (0.073% of total)
- Microsoft-phi-2: 9,175,040 trainable params (0.329% of total)

```bash
docker build -t lora_params .
docker run -it lora_params
```

## Key Features

- Multiple training approaches (DPO, PPO, SFT)
- Dataset generation for preference learning
- Customizable training parameters
- Docker support
- Performance metrics tracking
- LoRA parameter analysis

## Configuration Options

The training scripts support various configuration options:
- Model paths (teacher/student)
- Dataset generation parameters
- Training hyperparameters
- Batch sizes
- Learning rates
- Episode settings

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]
