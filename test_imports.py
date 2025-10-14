"""
Test script to verify module structure and imports.
This can be run after installing dependencies.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import data_utils
        print("✓ data_utils module found")
    except ImportError as e:
        print(f"✗ data_utils import failed: {e}")
    
    try:
        import models
        print("✓ models module found")
    except ImportError as e:
        print(f"✗ models import failed: {e}")
    
    try:
        import app
        print("✓ app module found")
    except ImportError as e:
        print(f"✗ app import failed: {e}")
    
    print("\nNote: You may see dependency errors if packages aren't installed yet.")
    print("Install dependencies with: pip install -r requirements.txt")

if __name__ == "__main__":
    test_imports()
