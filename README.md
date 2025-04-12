# Nvidia-Hackathon

Drive : https://drive.google.com/drive/folders/1Fynid6c0kF7EHdXoO69GGbrSxJEQXWz3?usp=drive_link

* This repository is the codebase of the Nvidia-Hackathon that me and my other team mate participated in during Oct 25 to Oct 27 2024.

Here is the simple description of the work we did:

Our goal was to enhance the accuracy of the Gemma 2B model for generating tags on the questions from the Law domain, for improved relevance search. We have used the Law Stack Exchange dataset from huggingface for this.

To achieve this, we utilized NVIDIA's NeMo Curator for faster data processing and NeMo microservices for fine tuning and evaluating the Large Language Model.

The data has a question, title, score and tags associated with it.

The key features that we have used for data processing from NeMo curator include:

* DocumentDataset: This allowed us to handle our dataset efficiently with Dask for scalable processing.

* UnicodeReformatter: We used this to encode the questions and titles into Unicode format.

* DocumentModifier: We have wrapped our custom text cleaning and section number formatting into the DocumentModifier functionality.

* PIIModifier: We have used this to remove personally identifiable information like names, phone numbers, addresses, locations and email addresses from questions and titles.

* We have manually investigated questions with 10, 20, 40, 50 words and found that questions with 15 or fewer words are not contextually complete and meaningful enough. So we have discarded all the questions with 15 or fewer words using WordCountFilter.

* ScoreFilter: We have used a score filter to exclude questions with a net upvote score below zero.

* We also utilized the RepeatingTopNGrams filter to eliminate questions that contained repetitive phrases, as these can introduce bias by overemphasizing certain terms and phrases. We have the n value of 5 and the repeating percentage as 15%.

* Further We have leveraged NeMo microservices to convert the downloaded Gemma 2B model into .nemo format so that the model is compatible with the NeMo framework and is also optimized to take advantage of Nvidia GPUs' capabilities.

* Our approach involved designing a custom system prompt that instructs the LLM what its task is, provided the question, title and the list of tags. We applied this to every question and title. 

* We used a 95% to 5% training-validation split for our dataset.

* In fine-tuning our model, we carefully selected our hyperparameters to balance performance, efficiency, and resource utilization.

* Our choice of bf16 precision strikes a balance between training speed and memory usage, crucial for handling our large language model. 

* The micro batch size of 1 and global batch size of 16 were chosen to optimize memory usage and gradient accumulation, allowing for stable training.

* We implemented a cosine annealing learning rate schedule with a base rate of 1e-6. This approach helps in finding an optimal convergence point by gradually reducing the learning rate, preventing overshooting in later stages of training.

* Infact, during our first trial of fine tuning the model, the training and validation loss are too far apart and also the validation is not converging while the training loss is, which is a clear sight of overfitting.

* So, we introduced weight decay of 0.01 was selected to prevent overfitting and ensure smooth gradient updates.

* We set our max steps to 2500 with evaluations every 200 steps. This frequent evaluation allowed us to monitor the model's performance closely and make necessary adjustments.

* We have used optimizer beta values of 0.9 and 0.95 to help achieve faster convergence and improved stability during the fine-tuning of large language models by adjusting the decay rates of the gradient and variance estimates in the Adam optimizer.

We have used virtual instances on Brev for this hackathon.

## Project Structure

```
.
├── Data_Processing.ipynb          # Data processing pipeline using NeMo Curator
├── Download_Model_Convert_to_NeMo.ipynb # Conversion of Gemma 2B to NeMo format
├── Hackathon Guidelines.md        # Official guidelines for the hackathon
├── Model_Training_Inference.ipynb # Model fine-tuning and inference pipeline
├── README.md                      # Project documentation
├── data/                          # Data directory
│   ├── curated_dataset/           # Processed and cleaned dataset
│   ├── raw/                       # Raw dataset files
│   ├── split/                     # Train/validation splits
│   ├── submission/                # Files for final submission
│   └── Tags and QA.csv            # Dataset with questions and tags
├── helpers/                       # Helper modules
│   ├── __init__.py                # Package initializer
│   ├── docbuilder.py              # Document building utilities
│   ├── filters.py                 # Data filtering functions
│   └── modifiers.py               # Text modification utilities
├── models/                        # Model files
│   └── gemma-2-2b.nemo            # Fine-tuned Gemma 2B model in NeMo format
├── results/                       # Training and evaluation results
│   ├── checkpoints/               # Saved model checkpoints
│   ├── inference/                 # Inference outputs
│   ├── cmd-args.log               # Command line arguments log
│   ├── events.out.tfevents        # TensorBoard events file
│   ├── git-info.log               # Git repository information
│   ├── hparams.yaml               # Hyperparameters configuration
│   ├── lightning_logs.txt         # PyTorch Lightning logs
│   ├── nemo_error_log.txt         # Error logs from NeMo
│   └── nemo_log_globalrank-0_localrank-0.txt # NeMo training logs
└── test_results_team_theta_hat.jsonl # Final evaluation results
```

## Folders

### Notebooks
- **`Data_Processing.ipynb`**: Implements data preprocessing using NeMo Curator, including text cleaning, PII removal, filtering by word count and score, and handling repetitive n-grams.
- **`Download_Model_Convert_to_NeMo.ipynb`**: Downloads the Gemma 2B model and converts it to NeMo format for compatibility with NVIDIA's NeMo framework.
- **`Model_Training_Inference.ipynb`**: Contains the fine-tuning pipeline and inference code using NeMo microservices.

### Helper Modules
- **`helpers/docbuilder.py`**: Utilities for building document structures for processing.
- **`helpers/filters.py`**: Implements custom filters for document processing including word count and score filters.
- **`helpers/modifiers.py`**: Contains text modification utilities for cleaning and formatting text data.

### Data
- **`data/raw/`**: Contains the original Law Stack Exchange dataset.
- **`data/curated_dataset/`**: Stores the processed and cleaned dataset ready for training.
- **`data/split/`**: Contains training and validation splits of the dataset.
- **`data/Tags and QA.csv`**: Consolidated dataset with questions, titles, and tags.

### Models
- **`models/gemma-2-2b.nemo`**: The fine-tuned Gemma 2B model optimized for tag generation in the legal domain.

### Results
- **`results/checkpoints/`**: Contains saved model checkpoints during training.
- **`results/inference/`**: Stores outputs from model inference.
- **`results/hparams.yaml`**: Configuration file with hyperparameters used for training.
- **`test_results_team_theta_hat.jsonl`**: Final evaluation results showing the model's performance.