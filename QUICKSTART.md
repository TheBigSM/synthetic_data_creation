# Quick Start Guide - 3-Phase Synthetic Data Pipeline

Complete step-by-step guide to generate synthetic COVID-19 content using our structured 3-phase approach.

## 🚀 Phase 1: Initial Setup & Testing (5 minutes)

### 1. Environment Setup

```bash
# Verify you're in the project directory
pwd  # Should show: .../synthetic_data_creation

# Check your API key configuration
cat .env | grep TOGETHER_API_KEY  # Should show your key

# Verify setup is working
python check_setup.py
```

### 2. Quick System Test

```bash
# Test basic functionality with a single item
python quick_start.py
```

**Expected Output:**
```
✅ Processing complete! Results saved to: results/quick_start_YYYYMMDD_HHMMSS.json
📊 Summary: Successfully processed 1 items
```

### 3. Launch Main Processing Notebook

```bash
# Open the comprehensive 3-phase processing notebook
jupyter notebook notebooks/data_generation/04_batch_processing.ipynb
```

## 📋 Phase 1: System Validation (10+10 items)

**In the notebook:**

1. **Run Setup Cells**: Execute all configuration and import cells
2. **Load Data**: Verify news articles and tweets are loaded correctly
3. **Process Phase 1**: Run the first processing section (10 articles + 10 tweets)
4. **Review Results**: Check the generated JSON files in `results/phase1_*.json`

**What to Look For:**
- ✅ All 20 items processed successfully
- ✅ Facts extracted correctly (2-4 facts per item typically)
- ✅ Modified facts are plausible but different from originals
- ✅ Synthetic content maintains readability

**If Issues Found:**
- Check API key configuration
- Verify data files are present
- Review error messages in notebook output
- Ensure sufficient API rate limit remaining

## 🔬 Phase 2: Quality Evaluation (100+100 items)

**Only proceed after Phase 1 success!**

### Step 1: Enable Phase 2 Processing

In the notebook, set:
```python
run_phase2 = True  # Change from False to True
```

Then execute the Phase 2 cell to process 100 articles + 100 tweets.

### Step 2: Manual Evaluation Setup

```bash
# Open manual evaluation notebook
jupyter notebook notebooks/evaluation/01_manual_evaluation.ipynb
```

**Manual Evaluation Process:**
1. **Sample Selection**: Creates evaluation samples from Phase 2 results
2. **Multi-Annotator Setup**: Configure for 3+ evaluators
3. **Rating Interface**: Rate items as "appropriate", "inappropriate", "in-between"
4. **Agreement Analysis**: Calculate Cohen's Kappa inter-annotator agreement

### Step 3: Automatic Evaluation

```bash
# Open automatic evaluation notebook  
jupyter notebook notebooks/evaluation/02_automatic_evaluation.ipynb
```

**Automatic Metrics:**
1. **Correctness**: How well modified facts were incorporated
2. **Coherence**: Text readability and linguistic quality
3. **Dissimilarity**: Semantic distance from original content

### Step 4: Quality Decision

**Proceed to Phase 3 only if:**
- ✅ Cohen's Kappa > 0.60 (substantial agreement)
- ✅ Automatic metrics show satisfactory scores
- ✅ Manual review indicates good quality

**If Quality is Insufficient:**
- κ 0.40-0.60: Improve annotation guidelines, re-evaluate
- κ < 0.40: Major methodology changes needed
- Return to configuration/prompting adjustments
## 🚀 Phase 3: Full Dataset Processing (700+700 items)

**Only proceed after successful Phase 2 evaluation!**

### Prerequisites Checklist
- ✅ Phase 2 completed successfully (100+100 items)
- ✅ Manual evaluation shows Cohen's Kappa > 0.60
- ✅ Automatic evaluation metrics are satisfactory
- ✅ ~2 hours available for uninterrupted processing
- ✅ Stable internet connection for API calls

### Enable Phase 3 Processing

In the main notebook, set:
```python
run_phase3 = True  # Change from False to True
```

**Processing Overview:**
- **Articles**: 700 items × ~6 seconds = ~70 minutes
- **Tweets**: 700 items × ~3 seconds = ~35 minutes  
- **Total Time**: ~105 minutes (~1.75 hours)
- **Total Cost**: ~$5.60 (1.4M tokens × $4/1M)

### Final Results

**Output Files:**
- `FINAL_news_articles_YYYYMMDD_HHMMSS.json` - Complete news results with metadata
- `FINAL_tweets_YYYYMMDD_HHMMSS.json` - Complete tweet results with metadata

**Results Structure:**
```json
{
  "metadata": {
    "processing_date": "20250724_150000",
    "phase": "Phase 3 - Full Dataset",
    "content_type": "news_articles",
    "total_items": 700,
    "successful_items": 695,
    "success_rate": 99.3
  },
  "results": [/* full synthetic data */]
}
```

## 📊 Understanding Your Results

### Data Quality Indicators

**Good Quality Results:**
- ✅ Success rate > 95%
- ✅ Average 2-4 facts extracted per item
- ✅ Modified facts are plausible but different
- ✅ Synthetic content maintains readability
- ✅ No API errors or rate limiting issues

### File Organization

```
results/
├── phase1_news_YYYYMMDD_HHMMSS.json      # Phase 1 testing (10 items)
├── phase1_tweets_YYYYMMDD_HHMMSS.json    # Phase 1 testing (10 items)
├── phase2_news_YYYYMMDD_HHMMSS.json      # Phase 2 evaluation (100 items)
├── phase2_tweets_YYYYMMDD_HHMMSS.json    # Phase 2 evaluation (100 items)
├── FINAL_news_articles_YYYYMMDD_HHMMSS.json  # Production results (700 items)
└── FINAL_tweets_YYYYMMDD_HHMMSS.json         # Production results (700 items)
```

### Next Steps After Processing

1. **Data Analysis**: Examine final results for patterns and quality
2. **ML Preparation**: Format data for classification model training
3. **Publication**: Document methodology and results for research
4. **Classification**: Train/test fake news detection models

## 🛠️ Troubleshooting

### Common Issues & Solutions

**Problem**: API rate limit errors
**Solution**: Processing includes automatic delays, but if issues persist, increase delays in config

**Problem**: Out of API credits
**Solution**: Check Together.ai account balance, add credits if needed

**Problem**: Low success rate (<90%)
**Solution**: Check data quality, verify prompts are working correctly

**Problem**: Notebook cells fail
**Solution**: Restart kernel, run cells in order, check all dependencies installed

**Problem**: Evaluation shows low agreement
**Solution**: Improve annotation guidelines, retrain evaluators, revise fact schemas

### Getting Help

1. **Check Logs**: Review notebook output for error details
2. **Verify Setup**: Run `python check_setup.py` again
3. **Data Issues**: Ensure CSV files have correct column names
4. **API Issues**: Verify Together.ai key and account status

## 🎯 Success Criteria

**By the end of this guide, you should have:**
- ✅ 700 synthetic news articles with modified facts
- ✅ 700 synthetic tweets with modified facts
- ✅ Comprehensive evaluation data showing quality metrics
- ✅ Results ready for classification model training
- ✅ Documentation of methodology and evaluation results

**Total Processing Time**: ~3 hours (setup + Phase 1 + Phase 2 evaluation + Phase 3)
**Total Cost**: ~$5.60 in API usage
**Output**: Complete synthetic dataset for misinformation research

---

*3-Phase Structured Approach • Quality-Controlled Processing • Research-Ready Results*
```

### Example Structure Preview:

```bash
# Quick peek at your data
python3 -c "
import pandas as pd
news = pd.read_csv('data/raw/recovery_news_prepared.csv')
tweets = pd.read_csv('data/raw/vaccination_all_tweets.csv')
print(f'News: {len(news)} articles, columns: {list(news.columns)}')
print(f'Tweets: {len(tweets)} tweets, columns: {list(tweets.columns)}')
print(f'Sample news: {news.iloc[0][\"content\"][:100]}...')
print(f'Sample tweet: {tweets.iloc[0][\"text\"]}')
"
```

## ⚡ Production Processing

### Small Batch Processing (Recommended)

```bash
# Process 5 news articles (will take ~15 minutes due to rate limits)
python3 -c "
from src.data_generation.pipeline import SyntheticDataPipeline
pipeline = SyntheticDataPipeline()
results = pipeline.run_full_pipeline(
    data_path='data/raw/recovery_news_prepared.csv',
    content_type='news',
    max_articles=5,
    llm_provider='together_llama4_maverick',
    max_facts_per_article=3
)
json_file = pipeline.save_results('json')
csv_file = pipeline.save_results('csv')
print(f'Processed {len(results)} articles')
print(f'Results: {json_file}, {csv_file}')
"
```

### Tweet Processing

```bash
# Process 3 tweets (safer batch size)
python3 -c "
from src.data_generation.pipeline import SyntheticDataPipeline
pipeline = SyntheticDataPipeline()
results = pipeline.run_full_pipeline(
    data_path='data/raw/vaccination_all_tweets.csv',
    content_type='social_media',
    max_articles=3,
    llm_provider='together_llama4_maverick',
    max_facts_per_article=3
)
json_file = pipeline.save_results('json')
print(f'Processed {len(results)} tweets -> {json_file}')
"
```

## 🎯 Rate Limit Management

### Your Current Limits:
- **0.6 queries per minute** (36 queries per hour)
- **Each article requires ~3 API calls** (extract + modify + generate)
- **Maximum: ~12 articles per hour**

### Batch Processing Strategy:

```bash
# Process 10 articles with automatic rate limiting
python3 -c "
import time
from src.data_generation.pipeline import SyntheticDataPipeline

def process_batch(start_idx, batch_size=2):
    pipeline = SyntheticDataPipeline()
    
    # Read data and slice
    import pandas as pd
    df = pd.read_csv('data/raw/recovery_news_prepared.csv')
    batch_df = df.iloc[start_idx:start_idx+batch_size]
    
    # Save temporary batch file
    batch_file = f'temp_batch_{start_idx}.csv'
    batch_df.to_csv(batch_file, index=False)
    
    # Process batch
    results = pipeline.run_full_pipeline(
        data_path=batch_file,
        content_type='news',
        max_articles=batch_size,
        llm_provider='together_llama4_maverick',
        max_facts_per_article=3
    )
    
    print(f'Batch {start_idx}: Processed {len(results)} articles')
    return results

# Process 6 articles in 3 batches of 2
for i in range(0, 6, 2):
    print(f'Processing batch starting at index {i}...')
    process_batch(i, 2)
    if i < 4:  # Don't wait after the last batch
        print('Waiting 3 minutes for rate limit...')
        time.sleep(180)  # 3 minutes between batches
"
```

## 📁 Understanding Output

### Output Directory Structure:

```bash
results/
├── synthetic_data/
│   ├── synthetic_data_pipeline_TIMESTAMP.json  # Detailed results
│   └── synthetic_data_pipeline_TIMESTAMP.csv   # Tabular format
└── evaluation_metrics/
    └── evaluation_template_TIMESTAMP.csv       # Quality assessment template
```

### Example Output Review:

```bash
# View your latest results
ls -la results/synthetic_data/ | tail -2

# Quick preview of CSV results
python3 -c "
import pandas as pd
import glob
latest_csv = sorted(glob.glob('results/synthetic_data/*.csv'))[-1]
df = pd.read_csv(latest_csv)
print(f'Latest results: {latest_csv}')
print(f'Columns: {list(df.columns)}')
print(f'Processed {len(df)} items')
if len(df) > 0:
    print(f'Sample synthetic text: {df.iloc[0][\"modified_article\"][:150]}...')
"
```

## 💰 Budget Tracking

### Cost Monitoring:

```bash
# Estimate costs for your data
python3 -c "
# Your current setup costs
cost_per_1k_tokens = 1.12  # Llama 4 Maverick
tokens_per_article = 1000  # Conservative estimate
articles_per_dollar = 1000 / (cost_per_1k_tokens)

print(f'💰 Budget Analysis:')
print(f'Cost per article: ~${cost_per_1k_tokens/1000:.4f}')
print(f'Articles per $1: ~{articles_per_dollar:.0f}')
print(f'Your 2029 news articles would cost: ~${2029 * cost_per_1k_tokens/1000:.2f}')
print(f'Processing time: ~{2029/12:.0f} hours at current rate limits')
"
```

## 🔧 Configuration Tweaking

### Switch Models (if needed):

```bash
# Edit config to use Scout model (cheaper but lower quality)
python3 -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['default_llm_provider'] = 'together_llama4_scout'
with open('config/config.yaml', 'w') as f:
    yaml.dump(config, f)
print('Switched to Scout model (cheaper)')
"
```

### Adjust Fact Limits:

```bash
# Process with fewer facts for faster/cheaper processing
python3 -c "
from src.data_generation.pipeline import SyntheticDataPipeline
pipeline = SyntheticDataPipeline()
results = pipeline.run_full_pipeline(
    data_path='data/raw/vaccination_all_tweets.csv',
    content_type='social_media',
    max_articles=1,
    llm_provider='together_llama4_maverick',
    max_facts_per_article=2  # Reduce from 3 to 2
)
print(f'Processed with 2 facts per item: {len(results)} results')
"
```

## 🚨 Troubleshooting

### Rate Limit Errors:
```bash
# If you see "rate limit" errors, wait longer:
python3 -c "import time; print('Waiting 2 minutes...'); time.sleep(120); print('Ready!')"
```

### Memory Issues:
```bash
# Process smaller batches
max_articles=1  # Process one at a time
```

### Check Pipeline Status:
```bash
# Verify everything is working
python3 -c "
from src.data_generation.pipeline import SyntheticDataPipeline
pipeline = SyntheticDataPipeline()
print('✅ Pipeline ready')
print(f'✅ Config loaded: {len(pipeline.config)} settings')
"
```

## 📋 Next Steps Checklist

- [ ] ✅ Tested single article/tweet processing
- [ ] ✅ Generated example files for review
- [ ] ✅ Processed first small batch (2-5 items)
- [ ] ⏭️ Review output quality in `examples/` directory
- [ ] ⏭️ Scale up to larger batches (10-20 items)
- [ ] ⏭️ Implement your own batch processing loop
- [ ] ⏭️ Create evaluation criteria for synthetic quality

## 🎯 Production Recommendations

1. **Start Small**: Always test with 1-2 items first
2. **Batch Wisely**: 2-5 items per batch to respect rate limits
3. **Monitor Costs**: Track API usage in Together.ai dashboard
4. **Save Frequently**: Pipeline saves automatically, but keep backups
5. **Quality Check**: Review examples before scaling up

---

**🎉 You're ready to generate synthetic COVID-19 data!** 

Questions? Check the examples in `examples/` directory or review the full README.md for detailed configuration options.
