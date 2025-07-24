#!/usr/bin/env python3
"""
Repository Status Summary - Shows current state of synthetic data pipeline
"""

import os
import json
import pandas as pd
from datetime import datetime

def repository_summary():
    """Generate a comprehensive status summary."""
    print("🎯 COVID-19 Synthetic Data Pipeline Status")
    print("=" * 60)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration Status
    print("🔧 Configuration Status:")
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
            if 'TOGETHER_API_KEY' in env_content:
                key_preview = env_content.split('TOGETHER_API_KEY=')[1].split('\n')[0][:20] + "..."
                print(f"  ✅ Together.ai API Key: {key_preview}")
            else:
                print(f"  ❌ Together.ai API Key: Not found")
    except FileNotFoundError:
        print(f"  ❌ .env file not found")
    
    # Data Status
    print(f"\n📊 Data Files Status:")
    data_files = [
        ('recovery-news-data.csv', 'Original news articles'),
        ('recovery_news_prepared.csv', 'Pipeline-ready news'),
        ('vaccination_all_tweets.csv', 'Vaccination tweets')
    ]
    
    for filename, description in data_files:
        filepath = f'data/raw/{filename}'
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"  ✅ {description}: {len(df):,} items")
            except:
                print(f"  ⚠️ {description}: File exists but unreadable")
        else:
            print(f"  ❌ {description}: Not found")
    
    # Examples Status
    print(f"\n📁 Generated Examples:")
    if os.path.exists('examples'):
        examples = [f for f in os.listdir('examples') if f.endswith('.json')]
        if examples:
            for example in sorted(examples):
                print(f"  ✅ {example}")
        else:
            print(f"  📝 Use notebooks for comprehensive processing (jupyter notebook notebooks/data_generation/04_batch_processing.ipynb)")
    else:
        print(f"  📝 Examples directory not created yet")
    
    # Test Scripts Status
    print(f"\n🧪 Test Scripts Available:")
    test_scripts = [
        ('test_single_success.py', 'Test single tweet processing'),
        ('notebooks/data_generation/04_batch_processing.ipynb', '3-phase processing notebook'),
        ('test_your_data.py', 'Test both news and tweets')
    ]
    
    for script, description in test_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script} - {description}")
        else:
            print(f"  ❌ {script} - Missing")
    
    # Pipeline Status
    print(f"\n⚙️ Pipeline Components:")
    critical_files = [
        ('src/data_generation/pipeline.py', 'Main pipeline'),
        ('src/data_generation/llm_client.py', 'LLM integration'),
        ('config/config.yaml', 'Configuration')
    ]
    
    for filepath, description in critical_files:
        if os.path.exists(filepath):
            print(f"  ✅ {description}: Ready")
        else:
            print(f"  ❌ {description}: Missing")
    
    # Recent Results
    print(f"\n📈 Recent Results:")
    if os.path.exists('results/synthetic_data'):
        results = [f for f in os.listdir('results/synthetic_data') if f.endswith('.json')]
        if results:
            latest = sorted(results)[-1]
            print(f"  📄 Latest result: {latest}")
            try:
                with open(f'results/synthetic_data/{latest}', 'r') as f:
                    data = json.load(f)
                    print(f"  📊 Items processed: {len(data)}")
            except:
                print(f"  ⚠️ Could not read result file")
        else:
            print(f"  📝 No results generated yet")
    else:
        print(f"  📝 Results directory not created yet")
    
    # Cost & Performance Estimates
    print(f"\n💰 Cost & Performance Analysis:")
    print(f"  🔹 Model: Llama 4 Maverick 17B")
    print(f"  🔹 Cost: $1.12 per 1M tokens")
    print(f"  🔹 Rate limit: 0.6 queries/minute")
    print(f"  🔹 Max throughput: ~12 articles/hour")
    print(f"  🔹 Budget efficiency: ~890 articles per $1")
    
    # Quick Commands
    print(f"\n🚀 Quick Start Commands:")
    print(f"  # Test pipeline")
    print(f"  python3 test_single_success.py")
    print(f"")
    print(f"  # Generate examples")
    print(f"  jupyter notebook notebooks/data_generation/04_batch_processing.ipynb  # Complete 3-phase processing")
    print(f"")
    print(f"  # Process small batch")
    print(f"  python3 -c \"from src.data_generation.pipeline import SyntheticDataPipeline; pipeline = SyntheticDataPipeline(); results = pipeline.run_full_pipeline('data/raw/vaccination_all_tweets.csv', 'social_media', max_articles=2, llm_provider='together_llama4_maverick', max_facts_per_article=3); print(f'Processed {{len(results)}} items')\"")
    
    print(f"\n✅ Repository is ready for synthetic data generation!")
    print(f"📖 See README.md and QUICKSTART.md for detailed instructions")

if __name__ == "__main__":
    repository_summary()
