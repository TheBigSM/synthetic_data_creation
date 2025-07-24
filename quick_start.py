#!/usr/bin/env python3
"""
Quick start script for COVID Synthetic Data Generation.
This script demonstrates the complete workflow with local data storage.
"""

import os
import sys
sys.path.append('.')

from dotenv import load_dotenv

def main():
    print("🚀 COVID Synthetic Data Generation - Quick Start")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check if any API key is configured
    api_keys = ["OPENAI_API_KEY", "TOGETHER_API_KEY"]
    configured_provider = None
    
    for key in api_keys:
        if os.getenv(key):
            if key == "TOGETHER_API_KEY":
                provider_name = "together_llama4_maverick"  # Use Maverick model
            else:
                provider_name = key.replace("_API_KEY", "").lower()
            configured_provider = provider_name
            print(f"✅ Found {key.replace('_API_KEY', '')} API key")
            break
    
    if not configured_provider:
        print("❌ No API keys configured!")
        print("\nPlease add at least one API key to your .env file:")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("TOGETHER_API_KEY=your_together_key_here")
        return
    
    # Import after environment check
    from src.data_generation.pipeline import SyntheticDataPipeline
    import pandas as pd
    
    print(f"\n🤖 Using {configured_provider.upper()} for LLM operations")
    
    # Create sample data if none exists
    sample_file = "data/raw/sample_covid_articles.csv"
    if not os.path.exists(sample_file):
        print(f"\n📝 Creating sample COVID data...")
        
        sample_articles = [
            {
                "content": "Health officials in New York announced that 150 patients have been diagnosed with COVID-19 variant B.1.1.7. The variant shows 70% increased transmissibility. Pfizer vaccine demonstrates 88% effectiveness against severe symptoms.",
                "source": "health_dept",
                "date": "2025-07-15"
            },
            {
                "content": "WHO researchers found that Omicron subvariant affects 25% more young adults. Moderna vaccine shows 92% effectiveness in preventing hospitalization in trials with 50,000 participants.",
                "source": "who_report", 
                "date": "2025-07-14"
            },
            {
                "content": "CDC reports 15% increase in COVID hospitalizations across metropolitan areas. Johnson & Johnson vaccine administered to 2 million people with 95% success rate.",
                "source": "cdc_report",
                "date": "2025-07-13"
            }
        ]
        
        os.makedirs("data/raw", exist_ok=True)
        df = pd.DataFrame(sample_articles)
        df.to_csv(sample_file, index=False)
        print(f"✅ Sample data created: {sample_file}")
    
    # Run the pipeline
    print(f"\n🔄 Running synthetic data generation pipeline...")
    print(f"📂 Input: {sample_file}")
    print(f"🤖 LLM Provider: {configured_provider}")
    print(f"📊 Processing 3 articles...")
    
    try:
        # Initialize pipeline
        pipeline = SyntheticDataPipeline()
        
        # Run complete pipeline
        results = pipeline.run_full_pipeline(
            data_path=sample_file,
            content_type="news",
            max_articles=3,
            llm_provider=configured_provider
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📈 Generated {len(results)} synthetic articles")
        
        # Save results locally
        print(f"\n💾 Saving results locally...")
        
        # Save in JSON format
        json_file = pipeline.save_results("json")
        print(f"✅ JSON results: {json_file}")
        
        # Save in CSV format  
        csv_file = pipeline.save_results("csv")
        print(f"✅ CSV results: {csv_file}")
        
        # Create evaluation template
        eval_file = pipeline.create_evaluation_template()
        print(f"✅ Evaluation template: {eval_file}")
        
        # Show sample result
        if results:
            print(f"\n📋 Sample Result Preview:")
            print(f"Original facts extracted: {len(results[0].extracted_facts)}")
            print(f"Modified facts: {len(results[0].modified_facts)}")
            print(f"Original length: {len(results[0].original_article)} chars")
            print(f"Synthetic length: {len(results[0].modified_article)} chars")
            
            if results[0].extracted_facts:
                print(f"\nFirst extracted fact:")
                fact = results[0].extracted_facts[0]
                print(f"  Type: {fact.get('name_of_fact', 'Unknown')}")
                print(f"  Data: {fact.get('specific_data', 'N/A')}")
        
        print(f"\n🎉 All data saved locally!")
        print(f"📁 Check these directories:")
        print(f"  • data/synthetic/ - Generated synthetic articles")
        print(f"  • results/evaluation_metrics/ - Evaluation templates")
        
        print(f"\n🔬 Next Steps:")
        print(f"1. Review generated data in data/synthetic/")
        print(f"2. Use evaluation template for manual annotation")
        print(f"3. Scale up with more articles")
        print(f"4. Train classification models")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print(f"Please check your API key and try again.")

if __name__ == "__main__":
    main()
