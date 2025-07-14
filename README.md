# Synthetic COVID News and Tweet Generation for Fake News Detection

## Project Overview

This project implements a structured 4-step methodology for creating synthetic COVID-related news articles and tweets using Large Language Models (LLMs), then training machine learning and deep learning models to classify real vs. fake news.

## 4-Step Methodology

### Step 1: Data Collection
- Real COVID news articles and tweets (already collected)
- Focus on vaccine-related content, statistics, medical effects

### Step 2: Fact Characterization  
Define COVID-specific fact types with:
- **Name of fact**: e.g., "Type", "Actor", "Location", "Statistics"
- **Description of fact**: What this fact represents
- **Common examples**: Typical values for this fact type

Example fact definition:
```json
{
  "name": "Type",
  "description": "The type of vaccine being discussed",
  "common_examples": "COVID vaccine, Pfizer-BioNTech, Moderna, Johnson & Johnson"
}
```

### Step 3: Fact Extraction
Extract structured facts from articles using LLM with predefined schema:
```json
{
  "name_of_fact": "Statistics",
  "description_of_fact": "Numerical data about vaccine effectiveness",
  "specific_data": "95% effectiveness against severe illness",
  "common_examples": "95% effectiveness, 150 patients, 50,000 participants"
}
```

### Step 4: Fact Manipulation
- Modify extracted facts to create plausible but false information
- Generate synthetic articles incorporating modified facts
- Maintain linguistic coherence and credibility

## Output Format

Each processed article generates:
```json
{
  "original_article": "Original news article text...",
  "extracted_facts": [
    {
      "name_of_fact": "Type",
      "description_of_fact": "The type of vaccine being discussed",
      "specific_data": "Pfizer-BioNTech COVID vaccine",
      "common_examples": "COVID vaccine, Pfizer-BioNTech, Moderna"
    }
  ],
  "modified_facts": [
    {
      "name_of_fact": "Type", 
      "specific_data": "Modified vaccine information"
    }
  ],
  "modified_article": "Synthetic article with false facts..."
}
```

## Project Structure

```
├── data/
│   ├── raw/                    # Original real COVID news articles and tweets
│   ├── processed/              # Cleaned and preprocessed data
│   ├── synthetic/              # Generated synthetic fake news and tweets
│   └── external/               # External datasets for benchmarking
├── notebooks/
│   ├── data_generation/        # Jupyter notebooks for LLM-based data generation
│   ├── classification/         # Notebooks for training classification models
│   └── evaluation/             # Model evaluation and comparison notebooks
├── src/
│   ├── data_generation/        # Scripts for synthetic data generation
│   ├── classification/         # Classification model implementations
│   └── utils/                  # Utility functions and helpers
├── models/
│   ├── saved_models/           # Trained model files
│   └── checkpoints/            # Training checkpoints
├── results/
│   ├── evaluation_metrics/     # Performance metrics and results
│   └── plots/                  # Visualization outputs
├── config/                     # Configuration files
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Evaluation Methodology

### Manual Evaluation
- 100-300 articles/tweets manually evaluated by 3+ annotators
- Rating categories: "appropriate", "inappropriate", "in-between"
- Calculate inter-annotator agreement using Cohen's Kappa
- Evaluate: fact extraction quality, fact modification quality, synthetic article quality

### Automatic Evaluation
1. **Correctness**: How well modified facts were incorporated
2. **Coherence**: Text readability and linguistic flow  
3. **Dissimilarity**: How different synthetic is from original

### Workflow
1. Start with 10 articles → manual evaluation
2. If successful → scale to 100 articles → evaluation  
3. If successful → process full dataset
4. Proceed to classification experiments

## Getting Started

### Prerequisites
- Python 3.8+
- LLM API key (OpenAI, Anthropic, or **Llama4**)
- Jupyter Notebook

### Supported LLM Providers
- **OpenAI** (GPT-3.5, GPT-4)
- **Anthropic** (Claude)
- **Llama4** ⭐ (New!)

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run structured methodology notebook
jupyter notebook notebooks/data_generation/03_structured_methodology.ipynb
```

### Key Notebooks
1. `03_structured_methodology.ipynb` - Complete 4-step pipeline
2. `01_covid_news_generation.ipynb` - News article processing
3. `02_covid_tweets_generation.ipynb` - Tweet processing

## Data Sources
- Real COVID news articles from reputable sources
- COVID-related tweets
- External fake news datasets for benchmarking

## Models to Evaluate
- Traditional ML: SVM, Random Forest, Naive Bayes
- Deep Learning: LSTM, BiLSTM, BERT, RoBERTa
- Tokenization: Word-level, subword (BPE), character-level

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrices
- Cross-dataset generalization performance

## Contributors
- Mateja Smiljanić
