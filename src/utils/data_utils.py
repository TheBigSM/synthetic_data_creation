"""
Utility functions for data loading and preprocessing.
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

def load_raw_data(data_type: str = "news") -> pd.DataFrame:
    """
    Load raw data from CSV files.
    
    Args:
        data_type: Either 'news' or 'tweets'
    
    Returns:
        DataFrame with loaded data
    """
    if data_type == "news":
        data_path = "data/raw/covid_news.csv"
    elif data_type == "tweets":
        data_path = "data/raw/covid_tweets.csv"
    else:
        raise ValueError("data_type must be 'news' or 'tweets'")
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        logging.warning(f"Data file {data_path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

def save_processed_data(df: pd.DataFrame, filename: str, data_type: str = "processed") -> None:
    """Save processed data to CSV."""
    output_dir = f"data/{data_type}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved to {output_path}")

def create_data_split(df: pd.DataFrame, test_size: float = 0.2, 
                     validation_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set
        validation_size: Fraction of data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df.get('label')
    )
    
    # Second split: train vs val
    val_size_adjusted = validation_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state, 
        stratify=train_val_df.get('label')
    )
    
    return train_df, val_df, test_df

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def ensure_dir_exists(dir_path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)
