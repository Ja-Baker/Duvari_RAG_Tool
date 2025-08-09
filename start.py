#!/usr/bin/env python3
"""
Startup script for Duvari RAG System
Provides detailed logging and environment validation
"""

import os
import sys

def validate_environment():
    """Validate required environment variables with Railway debugging"""
    print("🔍 Validating environment variables...")
    print(f"Total environment variables: {len(os.environ)}")
    
    # Show all environment variables for debugging
    print("All environment variables:")
    for key, value in sorted(os.environ.items()):
        if 'OPENAI' in key.upper():
            print(f"  {key}: {value[:20]}..." if len(value) > 20 else f"  {key}: {value}")
        elif key in ['PORT', 'PATH', 'PYTHON_VERSION'] or 'RAILWAY' in key.upper():
            print(f"  {key}: {value[:50]}..." if len(value) > 50 else f"  {key}: {value}")
    
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['DATA_FILE_PATH', 'PORT']
    
    # Try multiple ways to get environment variables
    openai_key_found = False
    for var in ['OPENAI_API_KEY', 'openai_api_key', 'OPENAI_KEY']:
        value = os.getenv(var) or os.environ.get(var)
        if value:
            print(f"✅ Found OpenAI key via {var}: {'*' * 20}...{value[-10:]}")
            openai_key_found = True
            break
    
    if not openai_key_found:
        print("❌ OPENAI_API_KEY: NOT FOUND in any variation")
        print("Available keys containing 'OPENAI':", [k for k in os.environ.keys() if 'OPENAI' in k.upper()])
        # Continue anyway to see what happens
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Using default")
    
    return True  # Continue even without OpenAI key for debugging

if __name__ == "__main__":
    print("🚀 Starting Duvari RAG System...")
    print("=" * 50)
    
    # Validate environment
    validate_environment()
    
    print("=" * 50)
    print("🚀 Starting Flask application with gunicorn...")
    
    # Use gunicorn for production
    import subprocess
    port = os.getenv('PORT', '5000')
    
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--timeout', '300',
        '--workers', '1',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'app:app'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)