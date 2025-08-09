#!/usr/bin/env python3
"""
Startup script for Duvari RAG System
Provides detailed logging and environment validation
"""

import os
import sys

def validate_environment():
    """Validate required environment variables"""
    print("🔍 Validating environment variables...")
    
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['DATA_FILE_PATH', 'PORT']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * 20}...{value[-10:]}")
        else:
            print(f"❌ {var}: NOT FOUND")
            return False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Using default")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Duvari RAG System...")
    print("=" * 50)
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed!")
        sys.exit(1)
    
    print("=" * 50)
    print("🎯 Environment validated successfully!")
    print("🚀 Starting Flask application...")
    
    # Import and run the main app
    from app import app
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)