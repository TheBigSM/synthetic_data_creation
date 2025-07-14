"""
Fact characterization schemas for different domains.
Step 2 of methodology: Define fact types with name, description, and common examples.
"""

from typing import List, Dict, Any

# COVID-specific fact schema based on methodology
COVID_FACT_SCHEMA = [
    {
        "name": "Type",
        "description": "The type of vaccine being discussed",
        "common_examples": "COVID vaccine, Pfizer-BioNTech, Moderna, Johnson & Johnson, AstraZeneca"
    },
    {
        "name": "Actor", 
        "description": "The entity that is offering or promoting the COVID vaccine (e.g. government, company, organization)",
        "common_examples": "Pfizer-BioNTech, Moderna, WHO, CDC, government health agencies"
    },
    {
        "name": "Location",
        "description": "Geographic location mentioned in relation to COVID vaccine",
        "common_examples": "New York City, United States, Europe, local hospitals, vaccination centers"
    },
    {
        "name": "Timeframe",
        "description": "Date, time period, or temporal reference mentioned",
        "common_examples": "January 2024, within two weeks, over 6 months, today, past week"
    },
    {
        "name": "Statistics",
        "description": "Numerical data, percentages, or quantitative measurements",
        "common_examples": "95% effectiveness, 150 patients, 50,000 participants, 25% increase"
    },
    {
        "name": "Medical_Effect",
        "description": "Health outcomes, side effects, or medical impacts mentioned",
        "common_examples": "severe illness protection, transmission reduction, side effects, immunity"
    },
    {
        "name": "Topic",
        "description": "The main COVID-related topic or focus of discussion",
        "common_examples": "vaccination campaign, variant detection, clinical trial, public health policy"
    }
]

# Tweet-specific schema (adapted for shorter content)
COVID_TWEET_FACT_SCHEMA = [
    {
        "name": "Type",
        "description": "The type of vaccine or COVID-related item mentioned",
        "common_examples": "COVID vaccine, Pfizer, Moderna, booster shot"
    },
    {
        "name": "Actor", 
        "description": "Entity mentioned in relation to COVID (person, organization, government)",
        "common_examples": "CDC, WHO, health officials, government, medical experts"
    },
    {
        "name": "Location",
        "description": "Geographic location mentioned",
        "common_examples": "NYC, London, local hospital, vaccination center"
    },
    {
        "name": "Statistics",
        "description": "Numbers, percentages, or quantitative data",
        "common_examples": "96% protection, 500 cases, 4th shot, 60% down"
    },
    {
        "name": "Medical_Effect",
        "description": "Health outcomes or medical impacts",
        "common_examples": "protection, side effects, recovery, immunity"
    },
    {
        "name": "Topic",
        "description": "Main COVID topic discussed",
        "common_examples": "vaccination, new variant, mandate, treatment"
    }
]

def get_fact_schema(content_type: str = "news") -> List[Dict[str, str]]:
    """
    Get appropriate fact schema based on content type.
    
    Args:
        content_type: Either 'news', 'tweets', or 'articles'
    
    Returns:
        List of fact definitions with name, description, and examples
    """
    if content_type.lower() in ['tweet', 'tweets']:
        return COVID_TWEET_FACT_SCHEMA
    else:
        return COVID_FACT_SCHEMA

def validate_fact_schema(schema: List[Dict]) -> bool:
    """Validate that fact schema has required fields."""
    required_fields = ['name', 'description', 'common_examples']
    
    for fact in schema:
        if not all(field in fact for field in required_fields):
            return False
    
    return True

def display_fact_schema(schema: List[Dict]) -> None:
    """Display fact schema in readable format."""
    print("FACT CHARACTERIZATION SCHEMA")
    print("=" * 40)
    
    for i, fact in enumerate(schema, 1):
        print(f"\nFact {i}: {fact['name']}")
        print(f"Description: {fact['description']}")
        print(f"Examples: {fact['common_examples']}")
    
    print("\n" + "=" * 40)
