# COVID-19 Synthetic Data Generation Pipeline

A structured research pipeline for generating synthetic COVID-19 content through fact manipulation methodology using Together.ai's Llama 4 models. Implements a **3-phase approach** with comprehensive evaluation framework.

## 🎯 Overview

This pipeline implements a **4-step structured methodology** with **3-phase processing approach**:

### 4-Step Core Methodology:
1. **Data Collection**: Load real COVID-19 news articles or tweets
2. **Fact Characterization**: Define structured fact schemas (actors, locations, statistics, etc.) 
3. **Fact Extraction**: Extract structured facts from original content using LLM
4. **Fact Manipulation**: Modify extracted facts and generate synthetic content

### 3-Phase Processing Approach:
- **Phase 1** (10+10 items): Initial system testing and validation
- **Phase 2** (100+100 items): Quality evaluation with manual and automatic metrics
- **Phase 3** (700+700 items): Full dataset processing after quality approval

## 🔧 Current Configuration

- **LLM Provider**: Together.ai 
- **Model**: Llama 4 Maverick 17B (meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
- **Cost**: $4.00 per 1M tokens (~$5.60 for full 1,400 items)
- **Rate Limit**: 100 requests/second  
- **Dataset**: 700 news articles + 700 tweets (selected from larger collections)
## 📊 Example Output Structure

### News Article Example
```json
{
  "generation_info": {
    "timestamp": "2025-07-24T14:21:18.826564",
    "content_type": "news",
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "max_facts_limit": 3
  },
  "original_content": "A novel respiratory virus that originated in Wuhan...",
  "extracted_facts": [
    {
      "name_of_fact": "Actor",
      "description_of_fact": "The entity that is offering or promoting the COVID vaccine",
      "specific_data": "Centers for Disease Control and Prevention"
    }
  ],
  "modified_facts": [
    {
      "name_of_fact": "Actor", 
      "description_of_fact": "The entity that is offering or promoting the COVID vaccine",
      "specific_data": "World Health Organization Europe"
    }
  ],
  "synthetic_content": "A novel respiratory virus that originated in Wuhan..."
}
```

### Tweet Example
```json
{
  "original_content": "Same folks said daikon paste could treat a cytokine storm #PfizerBioNTech",
  "extracted_facts": [
    {
      "name_of_fact": "Type",
      "specific_data": "Pfizer-BioNTech"
    }
  ],
  "modified_facts": [
    {
      "name_of_fact": "Type",
      "specific_data": "Moderna-NIH"
    }
  ],
  "synthetic_content": "Same folks said daikon paste could treat a cytokine storm #ModernaNIH"
}
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/navigate to project directory
cd synthetic_data_creation

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "TOGETHER_API_KEY=your_api_key_here" > .env

# Verify setup
python check_setup.py
```

### 2. Choose Your Approach

**Option A: Complete Notebook Workflow (Recommended)**
```bash
# Open main processing notebook
jupyter notebook notebooks/data_generation/04_batch_processing.ipynb

# Follow 3-phase approach:
# Phase 1: Test with 10+10 items → review results
# Phase 2: Process 100+100 items → comprehensive evaluation  
# Phase 3: Full 700+700 dataset → final production
```

**Option B: Quick Testing**
```bash
# Test single items (for development)
python quick_start.py  # Process 1-2 items quickly
```

### 3. Evaluation Framework

```bash
# Manual evaluation (after Phase 2)
jupyter notebook notebooks/evaluation/01_manual_evaluation.ipynb

# Automatic evaluation (after Phase 2)  
jupyter notebook notebooks/evaluation/02_automatic_evaluation.ipynb
```

## 📊 Data Sources & Scale

- **News Articles**: 700 selected from 2,029 total articles
- **Social Media**: 700 selected from 228,207 total tweets
- **Processing Time**: ~1.75 hours for full dataset
- **Estimated Cost**: ~$5.60 total

## 📈 3-Phase Processing Strategy

### Phase 1: System Validation (10+10 items)
- Quick test of pipeline functionality
- Verify fact extraction and modification quality  
- Check error handling and data flow
- **Purpose**: Catch setup issues early

### Phase 2: Quality Evaluation (100+100 items)
- Generate sufficient data for evaluation
- **Manual Evaluation**: 3+ annotators, inter-annotator agreement
- **Automatic Evaluation**: Correctness, coherence, dissimilarity metrics
- **Decision Point**: Proceed only if Cohen's Kappa > 0.60

### Phase 3: Production Processing (700+700 items)
- Full dataset processing after quality validation
- Comprehensive statistics and metadata
- Final results for classification experiments
- **Output**: Complete synthetic dataset ready for ML training

## � Evaluation Framework

### Manual Evaluation Requirements
- **Sample Size**: 100-300 items from Phase 2 results
- **Annotators**: 3+ evaluators for inter-annotator agreement
- **Categories**: Rate as "appropriate", "inappropriate", or "in-between"
- **Metrics**: Cohen's Kappa agreement score
- **Threshold**: κ > 0.60 required to proceed to Phase 3

### Automatic Evaluation Metrics
1. **Correctness**: How accurately modified facts were incorporated
2. **Coherence**: Text readability and linguistic flow quality
3. **Dissimilarity**: Semantic distance between original/synthetic content

### Quality Control Workflow
1. **Phase 1 Review**: Manual inspection of 10+10 results
2. **Phase 2 Evaluation**: Comprehensive evaluation with both methods
3. **Quality Gate**: Only proceed if evaluation scores meet thresholds
4. **Phase 3 Processing**: Full dataset generation after approval

## 🏗️ Project Structure

```
synthetic_data_creation/
├── notebooks/
│   ├── data_generation/
│   │   └── 04_batch_processing.ipynb      # Main 3-phase processing
│   └── evaluation/
│       ├── 01_manual_evaluation.ipynb     # Multi-annotator evaluation
│       └── 02_automatic_evaluation.ipynb  # Quantitative assessment
├── data/
│   └── raw/                               # Input datasets
│       ├── recovery_news_prepared.csv     # 2,029 articles (using 700)
│       └── vaccination_all_tweets.csv     # 228,207 tweets (using 700)
├── results/                               # Generated synthetic data
│   ├── phase1_*.json                      # Phase 1 test results
│   ├── phase2_*.json                      # Phase 2 evaluation data
│   └── FINAL_*.json                       # Phase 3 production results
├── src/
│   ├── data_generation/                   # Core pipeline components
│   │   ├── llm_client.py                 # Together.ai integration
│   │   ├── pipeline.py                   # Processing pipeline
│   │   └── fact_schemas.py               # Fact extraction schemas
│   └── utils/                            # Utility functions
├── config/                               # Configuration files
└── quick_start.py                        # Simple testing script
```

## 🔍 Fact Extraction Schema

The pipeline extracts 7 types of COVID-related facts:

1. **Type**: Vaccine types (Pfizer-BioNTech, Moderna, AstraZeneca, etc.)
2. **Actor**: Organizations (CDC, WHO, pharmaceutical companies, health officials)
3. **Location**: Geographic references (countries, states, cities, facilities)
4. **Timeframe**: Temporal information (dates, durations, deadlines)
5. **Statistics**: Numerical data (percentages, counts, rates, effectiveness)
6. **Medical_Effect**: Health outcomes (side effects, efficacy, symptoms)
7. **Topic**: Main discussion themes (policy, research, public health)

Each fact is structured as:
```json
{
  "name_of_fact": "Statistics",
  "description_of_fact": "Numerical data about vaccine effectiveness", 
  "specific_data": "95% effectiveness against severe illness",
  "common_examples": "95% effectiveness, 150 patients, 50,000 participants"
}
```

## 📖 Getting Started

### Prerequisites
- Python 3.8+
- Together.ai API key
- Jupyter Notebook
- ~2GB free disk space for results

### Installation
```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Configure API
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY

# 4. Verify setup
python check_setup.py

# 5. Start with notebooks
jupyter notebook
```

### Recommended Workflow
1. **Start**: Open `notebooks/data_generation/04_batch_processing.ipynb`
2. **Phase 1**: Test with 10+10 items, review results
3. **Phase 2**: Process 100+100 items, run evaluation notebooks
4. **Quality Check**: Ensure Cohen's Kappa > 0.60 before proceeding
5. **Phase 3**: Full 700+700 dataset processing
6. **Analysis**: Use results for classification model training

## 🎯 Research Applications

- **Misinformation Detection**: Train classifiers on synthetic false content
- **Content Analysis**: Study information manipulation patterns  
- **Robustness Testing**: Evaluate fact-checking systems
- **Educational Research**: Create controlled datasets for studies
- **ML Model Training**: Generate balanced datasets for classification

## ⚖️ Ethical Considerations

This tool is designed for research purposes. Generated synthetic content should:
- Be clearly labeled as synthetic/artificially generated
- Not be used to spread misinformation or deceive users
- Comply with research ethics guidelines and institutional review
- Respect platform terms of service and content policies
- Include appropriate disclaimers in research publications

---

*Research Pipeline • Together.ai Llama 4 Maverick • Structured Fact Manipulation • Handle Responsibly*

