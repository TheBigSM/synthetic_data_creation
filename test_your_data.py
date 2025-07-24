#!/usr/bin/env python3
"""
Test script to run the synthetic data pipeline with your actual CSV files.
"""

import os
import pandas as pd
from src.data_generation.pipeline import SyntheticDataPipeline

def prepare_news_data():
    """Prepare the recovery news data by mapping body_text to content column."""
    print("📰 Preparing recovery news data...")
    
    # Load original data
    df = pd.read_csv("data/raw/recovery-news-data.csv")
    print(f"Loaded {len(df)} news articles")
    
    # Map body_text to content for pipeline compatibility
    df_prepared = df.copy()
    df_prepared['content'] = df_prepared['body_text']
    
    # Save prepared data
    output_path = "data/raw/recovery_news_prepared.csv"
    df_prepared.to_csv(output_path, index=False)
    print(f"✅ Prepared news data saved to: {output_path}")
    
    return output_path

def test_small_sample(data_path, content_type, sample_name, max_articles=1):
    """Test pipeline with a very small sample."""
    print(f"\n🧪 Testing {sample_name} (sample of {max_articles})...")
    print("=" * 60)
    
    try:
        # Create pipeline
        pipeline = SyntheticDataPipeline()
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            data_path=data_path,
            content_type=content_type,
            max_articles=max_articles,
            llm_provider="together_llama4_maverick"  # Use your configured provider
        )
        
        print(f"✅ Successfully processed {len(results)} items")
        
        # Show first result
        if results:
            first_result = results[0]
            print(f"\n📄 Sample synthetic content preview:")
            print(f"Original facts: {len(first_result.extracted_facts) if hasattr(first_result, 'extracted_facts') else 'N/A'}")
            print(f"Modified facts: {len(first_result.modified_facts) if hasattr(first_result, 'modified_facts') else 'N/A'}")
            synthetic_text = first_result.synthetic_article if hasattr(first_result, 'synthetic_article') else ''
            print(f"Synthetic text: {synthetic_text[:200]}...")
            print(f"Label: {first_result.label if hasattr(first_result, 'label') else 'N/A'}")
        
        # Save results
        json_file = pipeline.save_results("json")
        csv_file = pipeline.save_results("csv")
        
        print(f"\n💾 Results saved:")
        print(f"- JSON: {json_file}")
        print(f"- CSV: {csv_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {sample_name}: {e}")
        return False

def main():
    """Main function to test both datasets."""
    print("🚀 Testing Synthetic Data Pipeline with Your Data")
    print("=" * 60)
    
    # Test 1: Recovery News Data
    try:
        news_path = prepare_news_data()
        news_success = test_small_sample(
            data_path=news_path,
            content_type="news", 
            sample_name="Recovery News Articles",
            max_articles=1  # Very small sample due to rate limits
        )
    except Exception as e:
        print(f"❌ News data test failed: {e}")
        news_success = False
    
    # Test 2: Vaccination Tweets Data (already has 'text' column)
    try:
        tweets_success = test_small_sample(
            data_path="data/raw/vaccination_all_tweets.csv",
            content_type="social_media",  # Use social media schema for tweets
            sample_name="Vaccination Tweets",
            max_articles=1  # Very small sample due to rate limits
        )
    except Exception as e:
        print(f"❌ Tweets data test failed: {e}")
        tweets_success = False
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"- Recovery News: {'✅ Success' if news_success else '❌ Failed'}")
    print(f"- Vaccination Tweets: {'✅ Success' if tweets_success else '❌ Failed'}")
    
    if news_success or tweets_success:
        print(f"\n🎉 At least one dataset is working! You can now:")
        print(f"- Process more articles by increasing max_articles")
        print(f"- Use the prepared data files for larger batches")
        print(f"- Monitor your Together.ai token usage carefully")

if __name__ == "__main__":
    main()
