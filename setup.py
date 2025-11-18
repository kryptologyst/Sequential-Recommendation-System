#!/usr/bin/env python3
"""
Setup script for Sequential Recommendation System.

This script helps set up the environment and generate initial data
for the sequential recommendation system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Sequential Recommendation System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "results", "logs", "models/checkpoints"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Generate sample data
    print("\n🔄 Generating sample data...")
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.data.data_pipeline import generate_sample_data, DataConfig
        
        # Generate data with reasonable size for demo
        config = DataConfig(
            num_users=200,
            num_items=50,
            min_sequence_length=3,
            max_sequence_length=20
        )
        
        generate_sample_data("data", config)
        print("✅ Sample data generated successfully")
        
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")
        print("You can generate data manually by running: python scripts/train.py")
    
    # Test the system
    print("\n🔄 Testing the system...")
    if run_command("python -c \"from src.models.sequential_models import MarkovChainModel; print('Models import successfully')\"", "Testing imports"):
        print("✅ System test passed")
    else:
        print("❌ System test failed")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py")
    print("2. Launch demo: streamlit run scripts/demo.py")
    print("3. Run tests: pytest tests/")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
