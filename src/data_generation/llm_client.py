"""
LLM client utilities for synthetic data generation.
"""

import openai
import os
import requests
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import time
import json

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

def clean_json_response(response_text: str) -> str:
    """Clean JSON response by removing markdown code blocks and handling truncated JSON."""
    # Remove markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3  
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    
    cleaned = response_text.strip()
    
    # Handle truncated JSON by trying to fix common issues
    if cleaned and not cleaned.endswith(']'):
        # Try to find the last complete object
        last_complete_brace = cleaned.rfind('}')
        if last_complete_brace != -1:
            # Check if we can add a closing bracket
            truncated_part = cleaned[last_complete_brace + 1:].strip()
            if not truncated_part or truncated_part.startswith(','):
                cleaned = cleaned[:last_complete_brace + 1] + ']'
    
    return cleaned

@dataclass
class FactExtractionResult:
    """Structure for structured fact extraction results."""
    original_text: str
    extracted_facts: List[Dict[str, Any]]  # Structured facts with name, description, specific_data
    confidence_scores: Optional[List[float]] = None

@dataclass
class SyntheticDataResult:
    """Structure for synthetic data generation results following the 4-step methodology."""
    original_article: str
    extracted_facts: List[Dict[str, Any]]  # Step 3: Facts with structured format
    modified_facts: List[Dict[str, Any]]   # Step 4: Modified facts before replacement  
    modified_article: str                   # Final synthetic article
    generation_metadata: Dict[str, Any] = None

class LLMClient:
    """Base class for LLM clients following structured fact methodology."""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
    
    def extract_structured_facts(self, text: str, fact_schema: List[Dict], max_facts: int = None) -> FactExtractionResult:
        """Extract structured facts using predefined schema."""
        raise NotImplementedError
    
    def modify_facts(self, extracted_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Modify facts to create plausible but false information."""
        raise NotImplementedError
    
    def generate_synthetic_article(self, original_text: str, modified_facts: List[Dict]) -> str:
        """Generate synthetic article with modified facts."""
        raise NotImplementedError

class OpenAIClient(LLMClient):
    """OpenAI API client implementing structured fact methodology."""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def extract_structured_facts(self, text: str, fact_schema: List[Dict], max_facts: int = None) -> FactExtractionResult:
        """Extract structured facts using COVID-specific schema."""
        
        # Create schema description for prompt
        schema_desc = ""
        for fact in fact_schema:
            schema_desc += f"- {fact['name']}: {fact['description']} (Examples: {fact['common_examples']})\n"
        
        prompt = f"""
        Extract facts from the following COVID-related text using the specified fact types. 
        For each fact found, provide:
        1. "name_of_fact": the category name
        2. "description_of_fact": description of what this fact represents  
        3. "specific_data": the exact value/information from the text
        4. "common_examples": similar examples for this fact type
        
        Fact Types to Look For:
        {schema_desc}
        
        Text: {text}
        
        Return ONLY a valid JSON array of facts found, like:
        [
            {{
                "name_of_fact": "Type",
                "description_of_fact": "The type of vaccine being discussed",
                "specific_data": "Pfizer-BioNTech COVID vaccine",
                "common_examples": "COVID vaccine, Pfizer-BioNTech, Moderna"
            }}
        ]
        
        Facts:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            facts_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                facts_json = json.loads(facts_text)
                if not isinstance(facts_json, list):
                    facts_json = []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON response: {facts_text}")
                facts_json = []
            
            # Limit facts if specified
            if max_facts and len(facts_json) > max_facts:
                facts_json = facts_json[:max_facts]
            
            return FactExtractionResult(
                original_text=text,
                extracted_facts=facts_json
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting structured facts: {e}")
            return FactExtractionResult(original_text=text, extracted_facts=[])
    
    def modify_facts(self, extracted_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Modify extracted facts to create false but plausible information."""
        if not extracted_facts:
            return []
        
        facts_json = json.dumps(extracted_facts, indent=2)
        
        prompt = f"""
        Modify the following extracted facts to create plausible but FALSE information for COVID-related content.
        Keep the same structure and fact types, but change the specific data to be incorrect.
        Make subtle but clearly false changes to numbers, dates, locations, vaccine types, etc.
        
        Original facts:
        {facts_json}
        
        Return ONLY a valid JSON array with the same structure but modified specific_data:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            modified_text = response.choices[0].message.content.strip()
            
            try:
                # Clean markdown code blocks if present
                clean_text = clean_json_response(modified_text)
                modified_facts = json.loads(clean_text)
                if not isinstance(modified_facts, list):
                    modified_facts = extracted_facts
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse modified facts JSON: {modified_text}")
                modified_facts = extracted_facts
            
            return modified_facts
            
        except Exception as e:
            self.logger.error(f"Error modifying facts: {e}")
            return extracted_facts
    
    def generate_synthetic_article(self, original_text: str, modified_facts: List[Dict]) -> str:
        """Generate synthetic article incorporating modified facts."""
        if not modified_facts:
            return original_text
        
        facts_to_incorporate = []
        for fact in modified_facts:
            facts_to_incorporate.append(f"- {fact.get('name_of_fact', 'Unknown')}: {fact.get('specific_data', 'N/A')}")
        
        facts_str = "\n".join(facts_to_incorporate)
        
        prompt = f"""
        Rewrite the following COVID-related article to incorporate these modified facts while maintaining:
        - Same writing style and tone
        - Natural flow and readability  
        - Credible but false information
        - Similar article length and structure
        
        Original article: {original_text}
        
        Modified facts to incorporate:
        {facts_str}
        
        Rewritten article:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic article: {e}")
            return original_text

class TogetherAIClient(LLMClient):
    """Together.ai API client implementing structured fact methodology."""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not TOGETHER_AVAILABLE:
            # Fallback to OpenAI-compatible client
            self.base_url = "https://api.together.xyz/v1"
            if self.api_key:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            else:
                self.client = None
        else:
            # Use native Together client
            if self.api_key:
                self.client = Together(api_key=self.api_key)
            else:
                self.client = None
    
    def extract_structured_facts(self, text: str, fact_schema: List[Dict], max_facts: int = None) -> FactExtractionResult:
        """Extract structured facts using COVID-specific schema with Together.ai."""
        
        # Create schema description for prompt
        schema_desc = ""
        for fact in fact_schema:
            schema_desc += f"- {fact['name']}: {fact['description']} (Examples: {fact['common_examples']})\n"
        
        prompt = f"""
        Extract facts from the following COVID-related text using the specified fact types. 
        For each fact found, provide:
        1. "name_of_fact": the category name
        2. "description_of_fact": description of what this fact represents  
        3. "specific_data": the exact value/information from the text
        4. "common_examples": similar examples for this fact type
        
        Fact Types to Look For:
        {schema_desc}
        
        Text: {text}
        
        Return ONLY a valid JSON array of facts found, like:
        [
            {{
                "name_of_fact": "Type",
                "description_of_fact": "The type of vaccine being discussed",
                "specific_data": "Pfizer-BioNTech COVID vaccine",
                "common_examples": "COVID vaccine, Pfizer-BioNTech, Moderna"
            }}
        ]
        
        Facts:
        """
        
        try:
            if not self.client:
                raise ValueError("Together.ai API key not configured")
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            facts_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean markdown code blocks if present
                clean_text = clean_json_response(facts_text)
                facts_json = json.loads(clean_text)
                if not isinstance(facts_json, list):
                    facts_json = []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON response: {facts_text}")
                facts_json = []
            
            # Limit facts if specified
            if max_facts and len(facts_json) > max_facts:
                facts_json = facts_json[:max_facts]
            
            return FactExtractionResult(
                original_text=text,
                extracted_facts=facts_json
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting structured facts with Together.ai: {e}")
            return FactExtractionResult(original_text=text, extracted_facts=[])
    
    def modify_facts(self, extracted_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Modify extracted facts to create false but plausible information."""
        if not extracted_facts:
            return []
        
        facts_json = json.dumps(extracted_facts, indent=2)
        
        prompt = f"""
        Modify the following extracted facts to create plausible but FALSE information for COVID-related content.
        Keep the same structure and fact types, but change the specific data to be incorrect.
        Make subtle but clearly false changes to numbers, dates, locations, vaccine types, etc.
        
        Original facts:
        {facts_json}
        
        Return ONLY a valid JSON array with the same structure but modified specific_data:
        """
        
        try:
            if not self.client:
                raise ValueError("Together.ai API key not configured")
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            modified_text = response.choices[0].message.content.strip()
            
            try:
                # Clean markdown code blocks if present
                clean_text = clean_json_response(modified_text)
                modified_facts = json.loads(clean_text)
                if not isinstance(modified_facts, list):
                    modified_facts = extracted_facts
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse modified facts JSON: {modified_text}")
                modified_facts = extracted_facts
            
            return modified_facts
            
        except Exception as e:
            self.logger.error(f"Error modifying facts with Together.ai: {e}")
            return extracted_facts
    
    def generate_synthetic_article(self, original_text: str, modified_facts: List[Dict]) -> str:
        """Generate synthetic article incorporating modified facts."""
        if not modified_facts:
            return original_text
        
        facts_to_incorporate = []
        for fact in modified_facts:
            facts_to_incorporate.append(f"- {fact.get('name_of_fact', 'Unknown')}: {fact.get('specific_data', 'N/A')}")
        
        facts_str = "\n".join(facts_to_incorporate)
        
        prompt = f"""
        Rewrite the following COVID-related article to incorporate these modified facts while maintaining:
        - Same writing style and tone
        - Natural flow and readability  
        - Credible but false information
        - Similar article length and structure
        
        Original article: {original_text}
        
        Modified facts to incorporate:
        {facts_str}
        
        Rewritten article:
        """
        
        try:
            if not self.client:
                raise ValueError("Together.ai API key not configured")
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic article with Together.ai: {e}")
            return original_text
    
def create_llm_client(provider: str = "together", **kwargs) -> LLMClient:
    """Factory function to create LLM clients."""
    provider_lower = provider.lower()
    
    # Handle provider naming scheme
    if provider_lower in ["openai", "openai_nano", "openai_mini", "openai_standard", "openai_premium"]:
        return OpenAIClient(**kwargs)
    elif provider_lower in ["together", "togetherai", "together_ai", "llama", "llama4"]:
        return TogetherAIClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openai variants, together variants")
