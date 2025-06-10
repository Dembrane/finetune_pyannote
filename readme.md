# Pyannote Model Fine-tuning

This repository contains tools and scripts for fine-tuning pyannote speaker segmentation models for improved speaker diarization performance.

## Overview

This project provides a workflow for:
- Creating custom datasets for speaker segmentation
- Training pyannote segmentation models on your data
- Integrating fine-tuned models into existing pyannote pipelines

## Installation

```bash
pip install pyannote.audio torch transformers
```

## Usage

### 1. Dataset Creation

Create your custom dataset following the guidelines from the diarizers repository:

```bash
# Follow the dataset creation guide
# https://github.com/huggingface/diarizers/tree/main/datasets
```

### 2. Model Training

Train your pyannote segmentation model using the diarizers training pipeline:

```bash
# Follow the training guide
# https://github.com/huggingface/diarizers/tree/main
```

### 3. Model Integration

Replace the segmentation model in your pyannote pipeline with your fine-tuned model:

```python
from pyannote.audio import Pipeline
from diarizers import SegmentationModel

# Load your fine-tuned model
model = SegmentationModel().from_pretrained("diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn")
model = model.to_pyannote_model()

# Initialize pipeline and replace segmentation model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline._segmentation.model = model.to(device)

# Use the pipeline with your custom model
diarization = pipeline("audio.wav")
```

## Resources

- [Dataset Creation Guide](https://github.com/huggingface/diarizers/tree/main/datasets)
- [Model Training Guide](https://github.com/huggingface/diarizers/tree/main)
- [pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)
