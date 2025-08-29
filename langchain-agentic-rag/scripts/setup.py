#!/usr/bin/env python3
"""Setup script for LangChain Agentic RAG system."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} found")
    
    # Check UV
    if not shutil.which("uv"):
        print("❌ UV not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    print("✅ UV package manager found")
    
    return True

def setup_environment():
    """Set up the environment file."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists():
        if env_example_path.exists():
            shutil.copy(env_example_path, env_path)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your Gemini API key")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["chroma_db", "logs", "docs", "tests"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def install_dependencies():
    """Install Python dependencies using UV."""
    return run_command("uv sync", "Installing dependencies")

def check_api_key():
    """Check if API key is configured."""
    env_path = Path(".env")
    
    if env_path.exists():
        with open(env_path) as f:
            content = f.read()
            if "your_gemini_api_key_here" in content:
                print("⚠️  Gemini API key not configured in .env file")
                print("Please set GEMINI_API_KEY in .env before running the system")
                return False
            elif "GEMINI_API_KEY=" in content:
                print("✅ Gemini API key appears to be configured")
                return True
    
    print("❌ Could not verify API key configuration")
    return False

def run_basic_tests():
    """Run basic system tests."""
    print("🧪 Running basic tests...")
    
    # Test imports
    test_imports = [
        "import backend.config",
        "import backend.models",
        "import backend.vector_store",
    ]
    
    for test_import in test_imports:
        try:
            result = subprocess.run([
                sys.executable, "-c", test_import
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"✅ Import test passed: {test_import}")
            else:
                print(f"❌ Import test failed: {test_import}")
                print(f"   Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Import test error: {e}")
            return False
    
    return True

def main():
    """Main setup function."""
    print("🤖 LangChain Agentic RAG System Setup")
    print("=====================================")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Setup failed - prerequisites not met")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run from project root directory")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Installing dependencies", install_dependencies),
    ]
    
    for description, func in steps:
        print(f"\n{description}...")
        if not func():
            print(f"\n❌ Setup failed at: {description}")
            sys.exit(1)
    
    # Check API key
    print("\n🔑 Checking API key configuration...")
    api_key_configured = check_api_key()
    
    # Run tests
    print("\n🧪 Running basic tests...")
    if run_basic_tests():
        print("✅ Basic tests passed")
    else:
        print("❌ Some tests failed, but setup can continue")
    
    # Final status
    print("\n" + "="*50)
    if api_key_configured:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the system: ./run.sh")
        print("2. Open http://localhost:8001 in your browser")
        print("3. Upload documents and start chatting!")
    else:
        print("⚠️  Setup completed with warnings")
        print("\nBefore running the system:")
        print("1. Edit .env file and set your GEMINI_API_KEY")
        print("2. Run the system: ./run.sh")
    
    print("\nFor help and documentation:")
    print("- README.md")
    print("- http://localhost:8001/docs (after starting)")

if __name__ == "__main__":
    main()