#!/usr/bin/env python3
"""
Simple setup verification for COVID Synthetic Data Generation project.
"""

import os
import sys

def main():
    print("🔍 COVID Synthetic Data Generation - Setup Check")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check key packages
    print("\n📦 Checking Dependencies...")
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing', 
        'openai': 'OpenAI API client',
        'anthropic': 'Anthropic API client',
        'requests': 'HTTP requests',
        'dotenv': 'Environment variables',
        'yaml': 'Configuration files',
        'tqdm': 'Progress bars'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            if package == 'dotenv':
                import dotenv
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check directories
    print("\n📁 Checking Directories...")
    directories = [
        "data/raw", "data/processed", "data/synthetic", 
        "results/evaluation_metrics", "models/saved_models"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"⚪ {directory} - Will be created automatically")
    
    # Check .env file
    print("\n🔑 Checking Environment...")
    if os.path.exists('.env'):
        print("✅ .env file exists")
        
        # Load and check API keys
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_keys = {
                "OPENAI_API_KEY": "OpenAI",
                "ANTHROPIC_API_KEY": "Anthropic", 
                "LLAMA4_API_KEY": "Llama4"
            }
            
            configured = []
            for key, provider in api_keys.items():
                if os.getenv(key):
                    print(f"✅ {provider} API key configured")
                    configured.append(provider)
                else:
                    print(f"⚪ {provider} API key not set")
            
            if not configured:
                print("\n⚠️  No API keys configured in .env file")
                print("Add at least one API key to start generating synthetic data")
                return False
            else:
                print(f"\n🎯 Ready to use: {', '.join(configured)}")
                
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
            return False
    else:
        print("❌ .env file not found")
        print("The .env file has been created - please add your API keys")
        return False
    
    # Check config file
    print("\n⚙️ Checking Configuration...")
    if os.path.exists('config/config.yaml'):
        print("✅ config/config.yaml exists")
    else:
        print("❌ config/config.yaml not found")
        return False
    
    # Test imports
    print("\n🧪 Testing Module Imports...")
    try:
        sys.path.append('.')
        from src.data_generation.pipeline import SyntheticDataPipeline
        from src.data_generation.fact_schemas import get_fact_schema
        from src.utils.data_utils import load_config
        print("✅ All modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Final status
    print("\n" + "=" * 60)
    print("🎉 Setup verification complete!")
    print("\n🚀 Ready to generate synthetic COVID data!")
    print("\nNext steps:")
    print("1. Add your API keys to .env file (if not done)")
    print("2. Run: python quick_start.py")
    print("3. Or open: notebooks/data_generation/03_structured_methodology.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Please fix the issues above before proceeding.")
        sys.exit(1)
