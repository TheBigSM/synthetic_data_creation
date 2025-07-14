"""
Main pipeline for synthetic data generation following the 4-step methodology:
1. Data collection (already done)
2. Fact characterization (define fact schemas)
3. Fact extraction (extract structured facts using LLM)
4. Fact manipulation (modify facts and generate synthetic articles)
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from tqdm import tqdm

from src.data_generation.llm_client import create_llm_client, SyntheticDataResult
from src.data_generation.fact_schemas import get_fact_schema, display_fact_schema
from src.utils.evaluation import EvaluationManager
from src.utils.data_utils import load_config, save_processed_data

class SyntheticDataPipeline:
    """
    Main pipeline implementing the 4-step methodology for synthetic data generation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.llm_client = None
        self.fact_schema = None
        self.results = []
        self.evaluation_manager = EvaluationManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def step1_data_collection(self, data_path: str) -> pd.DataFrame:
        """
        Step 1: Load collected data (already done in data collection phase).
        """
        self.logger.info("Step 1: Loading collected data...")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        self.logger.info(f"Loaded {len(df)} articles/tweets")
        return df
    
    def step2_fact_characterization(self, content_type: str = "news", custom_schema: List[Dict] = None):
        """
        Step 2: Define fact characterization schema with name, description, and examples.
        """
        self.logger.info("Step 2: Defining fact characterization schema...")
        
        if custom_schema:
            self.fact_schema = custom_schema
        else:
            self.fact_schema = get_fact_schema(content_type)
        
        display_fact_schema(self.fact_schema)
        self.logger.info(f"Defined {len(self.fact_schema)} fact types")
    
    def step3_fact_extraction(self, texts: List[str], max_facts_per_text: int = None, 
                             llm_provider: str = None) -> List[Dict]:
        """
        Step 3: Extract structured facts from texts using LLM.
        """
        self.logger.info("Step 3: Extracting structured facts...")
        
        if not self.llm_client:
            self._initialize_llm_client(llm_provider)
        
        if not self.fact_schema:
            raise ValueError("Must run step2_fact_characterization first")
        
        extraction_results = []
        
        for i, text in enumerate(tqdm(texts, desc="Extracting facts")):
            try:
                result = self.llm_client.extract_structured_facts(
                    text, 
                    self.fact_schema, 
                    max_facts_per_text
                )
                
                extraction_results.append({
                    'text_id': i,
                    'original_text': text,
                    'extracted_facts': result.extracted_facts
                })
                
            except Exception as e:
                self.logger.error(f"Error extracting facts from text {i}: {e}")
                extraction_results.append({
                    'text_id': i,
                    'original_text': text,
                    'extracted_facts': []
                })
        
        self.logger.info(f"Completed fact extraction for {len(extraction_results)} texts")
        return extraction_results
    
    def step4_fact_manipulation(self, extraction_results: List[Dict], 
                               llm_provider: str = None) -> List[SyntheticDataResult]:
        """
        Step 4: Modify facts and generate synthetic articles.
        """
        self.logger.info("Step 4: Modifying facts and generating synthetic articles...")
        
        if not self.llm_client:
            self._initialize_llm_client(llm_provider)
        
        synthetic_results = []
        
        for result in tqdm(extraction_results, desc="Generating synthetic data"):
            try:
                if not result['extracted_facts']:
                    self.logger.warning(f"No facts extracted for text {result['text_id']}, skipping...")
                    continue
                
                # Modify facts
                modified_facts = self.llm_client.modify_facts(result['extracted_facts'])
                
                # Generate synthetic article
                synthetic_text = self.llm_client.generate_synthetic_article(
                    result['original_text'], 
                    modified_facts
                )
                
                synthetic_result = SyntheticDataResult(
                    original_article=result['original_text'],
                    extracted_facts=result['extracted_facts'],
                    modified_facts=modified_facts,
                    modified_article=synthetic_text,
                    generation_metadata={
                        'text_id': result['text_id'],
                        'timestamp': datetime.now().isoformat(),
                        'model': self.llm_client.model_name,
                        'fact_schema_used': len(self.fact_schema)
                    }
                )
                
                synthetic_results.append(synthetic_result)
                
            except Exception as e:
                self.logger.error(f"Error generating synthetic data for text {result['text_id']}: {e}")
                continue
        
        self.results = synthetic_results
        self.logger.info(f"Generated {len(synthetic_results)} synthetic articles")
        return synthetic_results
    
    def run_full_pipeline(self, data_path: str, content_type: str = "news", 
                         max_articles: int = None, max_facts_per_article: int = None,
                         llm_provider: str = None) -> List[SyntheticDataResult]:
        """
        Run the complete 4-step pipeline.
        
        Args:
            data_path: Path to input data file
            content_type: Type of content ("news" or "tweets")
            max_articles: Maximum number of articles to process
            max_facts_per_article: Maximum facts to extract per article
            llm_provider: LLM provider to use ("openai", "anthropic", "llama4")
        """
        self.logger.info("Starting full synthetic data generation pipeline...")
        
        # Step 1: Data collection
        df = self.step1_data_collection(data_path)
        
        # Limit articles if specified
        if max_articles:
            df = df.head(max_articles)
            self.logger.info(f"Limited to {max_articles} articles for processing")
        
        # Extract text content
        if 'content' in df.columns:
            texts = df['content'].fillna('').tolist()
        elif 'text' in df.columns:
            texts = df['text'].fillna('').tolist()
        else:
            # Use first text column found
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                texts = df[text_columns[0]].fillna('').tolist()
            else:
                raise ValueError("No text column found in data")
        
        # Step 2: Fact characterization
        self.step2_fact_characterization(content_type)
        
        # Step 3: Fact extraction
        extraction_results = self.step3_fact_extraction(texts, max_facts_per_article, llm_provider)
        
        # Step 4: Fact manipulation
        synthetic_results = self.step4_fact_manipulation(extraction_results, llm_provider)
        
        self.logger.info("Pipeline completed successfully!")
        return synthetic_results
    
    def save_results(self, output_format: str = "json", output_dir: str = "data/synthetic") -> str:
        """Save pipeline results in specified format."""
        if not self.results:
            raise ValueError("No results to save. Run pipeline first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "json":
            # Save as JSON with full structure
            output_data = []
            for result in self.results:
                output_data.append({
                    'original_article': result.original_article,
                    'extracted_facts': result.extracted_facts,
                    'modified_facts': result.modified_facts,
                    'modified_article': result.modified_article,
                    'metadata': result.generation_metadata
                })
            
            filename = f"synthetic_data_pipeline_{timestamp}.json"
            filepath = f"{output_dir}/{filename}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        elif output_format.lower() == "csv":
            # Save as CSV for easier analysis
            output_data = []
            for result in self.results:
                # Add original article
                output_data.append({
                    'text': result.original_article,
                    'label': 'real',
                    'type': 'original',
                    'extracted_facts': json.dumps(result.extracted_facts),
                    'modified_facts': '',
                    'metadata': ''
                })
                
                # Add synthetic article  
                output_data.append({
                    'text': result.modified_article,
                    'label': 'fake',
                    'type': 'synthetic',
                    'extracted_facts': json.dumps(result.extracted_facts),
                    'modified_facts': json.dumps(result.modified_facts),
                    'metadata': json.dumps(result.generation_metadata)
                })
            
            df = pd.DataFrame(output_data)
            filename = f"synthetic_data_pipeline_{timestamp}.csv"
            filepath = f"{output_dir}/{filename}"
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'")
        
        self.logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def create_evaluation_template(self, output_file: str = None) -> str:
        """Create template for manual evaluation."""
        if not self.results:
            raise ValueError("No results to evaluate. Run pipeline first.")
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_template_{timestamp}.csv"
        
        # Convert results to evaluation format
        eval_data = []
        for result in self.results:
            eval_data.append({
                'original_article': result.original_article,
                'extracted_facts': result.extracted_facts,
                'modified_facts': result.modified_facts,
                'modified_article': result.modified_article
            })
        
        self.evaluation_manager.create_annotation_template(eval_data, output_file)
        return output_file
    
    def _initialize_llm_client(self, provider: str = None):
        """Initialize LLM client based on configuration."""
        llm_config = self.config.get('llm', {})
        
        # Use provided provider or default to first available
        if provider:
            selected_provider = provider
        else:
            # Default priority: llama4, openai, anthropic
            for prov in ['llama4', 'openai', 'anthropic']:
                if prov in llm_config:
                    selected_provider = prov
                    break
            else:
                selected_provider = 'openai'  # Final fallback
        
        if selected_provider in llm_config:
            config_params = llm_config[selected_provider].copy()
            
            # Handle Llama4 specific parameters
            if selected_provider == 'llama4':
                base_url = config_params.pop('base_url', None)
                self.llm_client = create_llm_client(
                    provider=selected_provider,
                    base_url=base_url,
                    **config_params
                )
            else:
                self.llm_client = create_llm_client(
                    provider=selected_provider,
                    **config_params
                )
            
            self.logger.info(f"Initialized {selected_provider} client with model {config_params.get('model_name', config_params.get('model'))}")
        else:
            raise ValueError(f"No configuration found for {selected_provider}. Available: {list(llm_config.keys())}")

def run_demo_pipeline(num_articles: int = 10, content_type: str = "news", 
                     llm_provider: str = None):
    """Run a demonstration of the pipeline with sample data."""
    
    # Create sample data
    sample_data = [
        {
            'content': 'Health officials in New York City announced today that a new variant of COVID-19 has been detected in 150 patients. The variant, designated B.1.1.7, appears to be 70% more transmissible than previous strains. Local hospitals have reported a 25% increase in admissions over the past week.',
        },
        {
            'content': 'A comprehensive study involving 50,000 participants shows that the latest COVID-19 vaccines maintain 95% effectiveness against severe illness. The research, conducted over 6 months, found that booster shots increase protection to 98% within two weeks of administration.',
        }
    ] * (num_articles // 2 + 1)  # Repeat to get desired number
    
    df = pd.DataFrame(sample_data[:num_articles])
    
    # Save sample data
    sample_file = "data/raw/sample_covid_news.csv"
    df.to_csv(sample_file, index=False)
    
    # Run pipeline
    pipeline = SyntheticDataPipeline()
    results = pipeline.run_full_pipeline(
        data_path=sample_file,
        content_type=content_type,
        max_articles=num_articles,
        llm_provider=llm_provider
    )
    
    # Save results
    json_file = pipeline.save_results("json")
    csv_file = pipeline.save_results("csv")
    
    # Create evaluation template
    eval_file = pipeline.create_evaluation_template()
    
    print(f"\nDemo pipeline completed!")
    print(f"- Generated {len(results)} synthetic articles")
    print(f"- Results saved to: {json_file}")
    print(f"- CSV format: {csv_file}")
    print(f"- Evaluation template: {eval_file}")
    
    return pipeline
