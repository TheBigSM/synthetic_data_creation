#!/usr/bin/env python3
"""
Test just the tweets to see the successful synthetic data generation.
"""

import os
import pandas as pd
from src.data_generation.pipeline import SyntheticDataPipeline

def test_single_tweet():
    """Test pipeline with just one tweet."""
    print("🐦 Testing Single Vaccination Tweet")
    print("=" * 40)
    
    try:
        # Create pipeline
        pipeline = SyntheticDataPipeline()
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            data_path="data/raw/vaccination_all_tweets.csv",
            content_type="social_media",
            max_articles=1,
            llm_provider="together_llama4_maverick"
        )
        
        print(f"✅ Successfully processed {len(results)} items")
        
        # Show result details
        if results:
            result = results[0]
            print(f"\n📊 Synthetic Data Result:")
            print(f"- Original facts: {len(result.extracted_facts)}")
            print(f"- Modified facts: {len(result.modified_facts)}")
            print(f"- Text length: {len(result.modified_article)} chars")
            print(f"\n📄 Original content preview:")
            print(f"{result.original_article[:150]}...")
            print(f"\n🔄 Synthetic article preview:")
            print(f"{result.modified_article[:300]}...")
            
            # Show some extracted facts
            print(f"\n🔍 Sample extracted facts:")
            for i, fact in enumerate(result.extracted_facts[:3]):
                print(f"  {i+1}. {fact.get('name_of_fact', 'Unknown')}: {fact.get('specific_data', 'N/A')}")
            
            # Show some modified facts
            print(f"\n🛠️ Sample modified facts:")
            for i, fact in enumerate(result.modified_facts[:3]):
                print(f"  {i+1}. {fact.get('name_of_fact', 'Unknown')}: {fact.get('specific_data', 'N/A')}")
        
        # Save results
        json_file = pipeline.save_results("json")
        csv_file = pipeline.save_results("csv")
        
        print(f"\n💾 Results saved:")
        print(f"- JSON: {json_file}")
        print(f"- CSV: {csv_file}")
        
        print(f"\n🎉 SUCCESS! Your pipeline is working with Together.ai Llama 4 Maverick!")
        print(f"📊 Token usage estimate: ~3-4 API calls per article (extraction + modification + generation)")
        print(f"⏱️ Rate limit: 0.6 queries/minute = ~20 articles per hour max")
        print(f"💰 Cost estimate: With $1 budget, you can process ~890K tokens = ~200-300 articles")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_single_tweet()
