"""
LLM client utilities for synthetic data generation.
"""

import openai
import anthropic
import os
import requests
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import time
import json

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
                modified_facts = json.loads(modified_text)
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

class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def extract_facts(self, text: str, num_facts: int = 3) -> FactExtractionResult:
        """Extract key facts using Claude."""
        # Similar implementation to OpenAI but using Anthropic's API
        # Implementation would go here
        pass
    
    def generate_synthetic_text(self, original_text: str, modified_facts: List[str]) -> str:
        """Generate synthetic text using Claude."""
        # Implementation would go here
        pass

class Llama4Client(LLMClient):
    """Llama4 API client implementing structured fact methodology."""
    
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("LLAMA4_API_KEY")
        self.base_url = base_url or os.getenv("LLAMA4_BASE_URL", "https://api.llama4.ai/v1")
        
        # If using OpenAI-compatible API, we can use OpenAI client
        if self.base_url and self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
    
    def extract_structured_facts(self, text: str, fact_schema: List[Dict], max_facts: int = None) -> FactExtractionResult:
        """Extract structured facts using COVID-specific schema with Llama4."""
        
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
            if self.client:
                # Use OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                facts_text = response.choices[0].message.content.strip()
            else:
                # Direct HTTP API call
                facts_text = self._make_direct_api_call(prompt)
            
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
            self.logger.error(f"Error extracting structured facts with Llama4: {e}")
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
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                modified_text = response.choices[0].message.content.strip()
            else:
                modified_text = self._make_direct_api_call(prompt)
            
            try:
                modified_facts = json.loads(modified_text)
                if not isinstance(modified_facts, list):
                    modified_facts = extracted_facts
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse modified facts JSON: {modified_text}")
                modified_facts = extracted_facts
            
            return modified_facts
            
        except Exception as e:
            self.logger.error(f"Error modifying facts with Llama4: {e}")
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
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            else:
                return self._make_direct_api_call(prompt)
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic article with Llama4: {e}")
            return original_text
    
    def _make_direct_api_call(self, prompt: str) -> str:
        """Make direct HTTP API call for non-OpenAI compatible endpoints."""
        if not self.base_url or not self.api_key:
            raise ValueError("Llama4 API URL and key must be configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("text", "")
        else:
            raise Exception(f"Llama4 API error: {response.status_code} - {response.text}")

def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """Factory function to create LLM clients."""
    provider_lower = provider.lower()
    
    # Handle new provider naming scheme
    if provider_lower in ["openai", "openai_nano", "openai_mini", "openai_standard", "openai_premium"]:
        return OpenAIClient(**kwargs)
    elif provider_lower in ["anthropic", "anthropic_haiku", "anthropic_sonnet"]:
        return AnthropicClient(**kwargs)
    elif provider_lower in ["llama4", "groq_llama4", "fireworks_llama4"]:
        return Llama4Client(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openai variants, anthropic variants, llama4 variants")
