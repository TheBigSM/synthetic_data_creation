#!/bin/bash
# Activation script for COVID Synthetic Data Generation project
# Usage: source activate_env.sh

echo "🚀 Activating COVID Synthetic Data Generation Environment"
echo "=================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "🐍 Python: $(python --version)"
    echo "📦 Pip: $(pip --version | cut -d' ' -f1-2)"
else
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    return 1
fi

# Check if in correct directory
if [ ! -f "config/config.yaml" ]; then
    echo "⚠️  Warning: Not in project root directory"
    echo "Please navigate to the synthetic_data_creation directory"
fi

# Show available commands
echo ""
echo "🔧 Available Commands:"
echo "  python check_setup.py          - Verify setup"
echo "  python quick_start.py          - Quick start demo"
echo "  jupyter notebook               - Open notebooks"
echo ""
echo "📁 Key Files:"
echo "  .env                          - Add your API keys here"
echo "  config/config.yaml            - Configuration settings"
echo "  quick_start.py                - Simple demo script"
echo ""
echo "🎯 Ready to generate synthetic COVID data!"
