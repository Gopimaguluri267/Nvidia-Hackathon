# ODSC West 2024 Hackathon with NVIDIA

Welcome to the central repository for the ODSC West 2024 Hackathon with NVIDIA!

❔ For more information on the hackathon itself, check out [this webpage](https://odsc.com/california/hackathon/) or [this FAQ](https://docs.google.com/document/d/1W_ClcEYBsNeLvDZSYImYG1CrrSTSj7Fo_ZOE_Ev1D1k/edit?tab=t.0#heading=h.8eu9ji7smdv6). ❔

## Overview of the Hackathon:

Your goal in this Hackathon is to train `google/gemma-2-2b` using PEFT LoRA on a legal tag-classification task. You'll be using the [Law-StackExchange](https://huggingface.co/datasets/ymoslem/Law-StackExchange) as the base dataset for this task.

You will use [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) to curate data and [NeMo FW](https://github.com/NVIDIA/NeMo) to customize it and then evaluate your model!

You are free to: 
- Modify training hyperparameters
- Modify, Augment (with SDG, etc) the training dataset
- Modify the NeMo Curator curation pipeline

### Scoring:

Your (or your team's) scores will be based on multi-label F1 scores, determined by comparing your generated predictions on the submission dataset against the held-out labels. 

In case of the ties, we will be using the videos you submitted to gauge your understanding of the data, NeMo curator, and NeMo framework:
- Understanding the data and usage of the NeMo Curator
- Deep understanding of data processing pipeline. Usage of the most relevant data processing steps.
- Understanding the fine-tuning and usage of the NeMo framework	
- Excellent grasp of fine-tuning techniques and using various hyperparameters for optimal model accuracy and customization.

## Overview of this Repository

The repository will guide you through a boilerplate example of NeMo Curator curation pipelines and NeMo FW customization, model loading, and inference.

There are a total of three Jupyter Notebooks to work through:

1. [Data Curation](https://github.com/chrisalexiuk-nvidia/ODSC-Hackathon-Repository/blob/main/step-1-data-curation.ipynb)
  - This notebook will take you through the downloading, processing, and then curating the target dataset
2. [Downloading the Model](https://github.com/chrisalexiuk-nvidia/ODSC-Hackathon-Repository/blob/main/step-2-download-model.ipynb)
  - This notebook will download the model and convert it to a NeMo FW compatible format
3. [Training a LoRA Adapter](https://github.com/chrisalexiuk-nvidia/ODSC-Hackathon-Repository/blob/main/step-3-training.ipynb)
  - This notebook will go through how to fine-tune the model using PEFT LoRA, and then how to generate submission responses

## Deliverables

You must submit (according to [this form](https://docs.google.com/forms/d/e/1FAIpQLSf3zeyPKXqs6Bhe_wmPRA-fzKncm_aNlliZAOMIsW7NUYr6kg/viewform)) in a *SINGLE* Google Drive:

1. Your predicted tag submission `.JSONL` file.
2. Your LoRA Adapters
3. Your notebooks (with outputs)
4. A 3min. video explaining your process (code walkthrough not required).

## Conclusion

Have fun! 🎉



