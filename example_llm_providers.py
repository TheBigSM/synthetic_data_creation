#!/usr/bin/env python3
"""
Example script demonstrating how to use different LLM providers
with the synthetic data generation pipeline.
"""

import os
import sys
sys.path.append('.')

from src.data_generation.pipeline import SyntheticDataPipeline
from src.utils.data_utils import load_config

def main():
    """Demo script showing how to use different LLM providers."""
    
    print("🤖 LLM Provider Demo for COVID Synthetic Data Generation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Check available providers
    available_providers = []
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("openai")
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found")
    
    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
        print("✅ Anthropic API key found")
    else:
        print("❌ Anthropic API key not found")
    
    # Check Llama4
    if os.getenv("LLAMA4_API_KEY"):
        available_providers.append("llama4")
        print("✅ Llama4 API key found")
    else:
        print("❌ Llama4 API key not found")
    
    if not available_providers:
        print("\n❌ No LLM providers configured!")
        print("Please add at least one API key to your .env file:")
        print("- OPENAI_API_KEY=your_openai_key")
        print("- ANTHROPIC_API_KEY=your_anthropic_key")
        print("- LLAMA4_API_KEY=your_llama4_key")
        return
    
    print(f"\n🎯 Available providers: {', '.join(available_providers)}")
    
    # Sample COVID article for testing
    sample_article = """
    Health officials announced today that a new COVID-19 variant has been detected 
    in 250 patients across New York City. The variant, designated XB.1.5, shows 
    increased transmissibility compared to previous strains. Pfizer-BioNTech vaccine 
    demonstrates 88% effectiveness against severe symptoms from this variant. 
    Local hospitals report a 15% increase in admissions this week.
    """
    
    # Test each available provider
    for provider in available_providers:
        print(f"\n{'='*20} Testing {provider.upper()} {'='*20}")
        
        try:
            # Initialize pipeline with specific provider
            pipeline = SyntheticDataPipeline()
            pipeline._initialize_llm_client(provider)
            
            # Step 2: Fact characterization (COVID-specific)
            pipeline.step2_fact_characterization("news")
            
            # Step 3: Extract facts
            print(f"🔍 Extracting facts with {provider}...")
            extraction_results = pipeline.step3_fact_extraction([sample_article], 
                                                               max_facts_per_text=3,
                                                               llm_provider=provider)
            
            if extraction_results and extraction_results[0]['extracted_facts']:
                print(f"✅ Extracted {len(extraction_results[0]['extracted_facts'])} facts")
                
                # Show extracted facts
                for i, fact in enumerate(extraction_results[0]['extracted_facts'][:2]):
                    print(f"  Fact {i+1}: {fact.get('name_of_fact', 'Unknown')} = {fact.get('specific_data', 'N/A')}")
                
                # Step 4: Generate synthetic article
                print(f"🔄 Generating synthetic article with {provider}...")
                synthetic_results = pipeline.step4_fact_manipulation(extraction_results,
                                                                   llm_provider=provider)
                
                if synthetic_results:
                    print(f"✅ Generated synthetic article ({len(synthetic_results[0].modified_article)} chars)")
                    print(f"   Modified {len(synthetic_results[0].modified_facts)} facts")
                else:
                    print("❌ Failed to generate synthetic article")
            else:
                print("❌ No facts extracted")
                
        except Exception as e:
            print(f"❌ Error with {provider}: {e}")
    
    print(f"\n🎉 Demo completed! Try the main notebook for full pipeline.")
    print("📝 Next steps:")
    print("1. Run: jupyter notebook notebooks/data_generation/03_structured_methodology.ipynb")
    print("2. Choose your preferred LLM provider in the notebook")
    print("3. Process your COVID dataset!")

if __name__ == "__main__":
    main()
