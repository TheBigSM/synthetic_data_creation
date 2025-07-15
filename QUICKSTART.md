# Quick Start Guide

## 🚀 Getting Started with COVID Synthetic Data Generation

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/TheBigSM/synthetic_data_creation.git
cd synthetic_data_creation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify setup
python check_setup.py

# Setup environment variables
# The .env file already exists - just add your API keys
nano .env  # or use your preferred editor
```

### Supported LLM Providers
- **OpenAI**: Add `OPENAI_API_KEY=your_key_here` to .env
- **Anthropic**: Add `ANTHROPIC_API_KEY=your_key_here` to .env  
- **Llama4** ⭐: Add `LLAMA4_API_KEY=your_key_here` and `LLAMA4_BASE_URL=your_endpoint` to .env

### 2. Run the 4-Step Methodology

#### Option A: Complete Pipeline (Recommended)
```bash
jupyter notebook notebooks/data_generation/03_structured_methodology.ipynb
```
This notebook implements the complete 4-step methodology:
- Step 1: Data collection
- Step 2: Fact characterization  
- Step 3: Fact extraction
- Step 4: Fact manipulation

#### Option B: Individual Components
```bash
# For news articles
jupyter notebook notebooks/data_generation/01_covid_news_generation.ipynb

# For tweets
jupyter notebook notebooks/data_generation/02_covid_tweets_generation.ipynb
```

### 3. Expected Workflow

1. **Start Small**: Process 10 articles first
2. **Manual Evaluation**: Use generated evaluation template
3. **Scale Up**: If successful, process 100 articles
4. **Full Dataset**: Process complete dataset
5. **Classification**: Move to ML/DL model training

### 4. Output Format

Each processed article generates JSON with:
```json
{
  "original_article": "Original text...",
  "extracted_facts": [
    {
      "name_of_fact": "Type",
      "description_of_fact": "The type of vaccine being discussed", 
      "specific_data": "Pfizer-BioNTech COVID vaccine"
    }
  ],
  "modified_facts": [...],
  "modified_article": "Synthetic text with false facts..."
}
```

### 5. Evaluation

- **Manual**: Share CSV template with 3+ annotators
- **Automatic**: Correctness, coherence, dissimilarity metrics
- **Agreement**: Cohen's Kappa for inter-annotator reliability

### 6. File Structure

- `data/synthetic/` - Generated synthetic articles
- `results/` - Evaluation templates and metrics
- `src/` - Core pipeline and utilities
- `config/config.yaml` - Model and processing settings

### 7. Troubleshooting

**No API Key**: Add your LLM API key to `.env` file:
- OpenAI: `OPENAI_API_KEY=your_key_here`
- Anthropic: `ANTHROPIC_API_KEY=your_key_here`  
- Llama4: `LLAMA4_API_KEY=your_key_here` and `LLAMA4_BASE_URL=your_endpoint`

**Import Errors**: Make sure you're running from the project root directory

**No Facts Extracted**: Check your input data format and LLM model settings

**Llama4 Connection Issues**: Verify your `LLAMA4_BASE_URL` endpoint is correct

---

For detailed methodology and implementation details, see the main [README.md](README.md).
