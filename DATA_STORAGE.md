# Data Directory Structure

This file documents the local data storage structure for the COVID synthetic data generation project.

## 📁 Directory Structure

```
data/
├── raw/                        # Original real COVID data
│   ├── covid_news/            # COVID news articles
│   ├── covid_tweets/          # COVID tweets  
│   └── external/              # External datasets
├── processed/                  # Cleaned and preprocessed data
│   ├── cleaned_articles.csv   # Processed news articles
│   ├── cleaned_tweets.csv     # Processed tweets
│   └── train_test_splits/     # Data splits for ML training
├── synthetic/                  # Generated synthetic data
│   ├── fake_articles.json     # Synthetic news articles
│   ├── fake_tweets.json       # Synthetic tweets
│   └── generation_metadata/   # Generation logs and metadata
└── external/                   # External benchmarking datasets

results/
├── evaluation_metrics/         # Performance metrics and results
│   ├── manual_annotations.csv # Human evaluation results
│   ├── automatic_metrics.json # Automated evaluation scores
│   └── evaluation_reports/    # Detailed evaluation reports
└── plots/                     # Visualization outputs
    ├── performance_charts/    # Model performance plots
    └── data_analysis/         # Data exploration plots

models/
├── saved_models/              # Trained classification models
│   ├── traditional_ml/       # SVM, Random Forest, etc.
│   └── deep_learning/        # BERT, RoBERTa, custom models
└── checkpoints/              # Training checkpoints
```

## 🔧 Automatic Directory Creation

The pipeline automatically creates these directories when:
- Saving synthetic data: `data/synthetic/`
- Saving results: `results/evaluation_metrics/`
- Saving models: `models/saved_models/`

## 📊 Data File Formats

### Synthetic Data Output
- **JSON Format**: Complete structured data with metadata
- **CSV Format**: Tabular format for analysis
- **Evaluation Templates**: CSV files for manual annotation

### Example Output Files
- `data/synthetic/covid_synthetic_YYYYMMDD_HHMMSS.json`
- `data/synthetic/covid_synthetic_YYYYMMDD_HHMMSS.csv`
- `results/evaluation_metrics/evaluation_template_YYYYMMDD.csv`

## 🚀 Quick Usage

```python
# The pipeline automatically saves to these local directories
pipeline = SyntheticDataPipeline()
results = pipeline.run_full_pipeline("your_data.csv")

# Results automatically saved to:
# - data/synthetic/ (JSON and CSV formats)
# - results/evaluation_metrics/ (evaluation template)
```

All data is stored locally by default - no cloud storage required!
