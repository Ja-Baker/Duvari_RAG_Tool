from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
import time
import hashlib

# Import OpenAI with debugging
try:
    from openai import OpenAI
    import openai
    print(f"✅ OpenAI library imported successfully - version: {openai.__version__}")
except ImportError as e:
    print(f"❌ Failed to import OpenAI library: {e}")
    OpenAI = None

# Try to import numpy, fall back to math if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  NumPy not available, using math library for calculations")
    NUMPY_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Railway environment variable debugging
print("🔍 Debugging environment variables...")
print(f"Total environment variables: {len(os.environ)}")
print("All environment variables:")
for key, value in os.environ.items():
    if 'OPENAI' in key.upper():
        print(f"  {key}: {value[:20]}..." if len(value) > 20 else f"  {key}: {value}")
    elif key in ['PORT', 'RAILWAY_', 'PATH']:
        print(f"  {key}: {value[:50]}..." if len(value) > 50 else f"  {key}: {value}")
    elif 'PROXY' in key.upper():
        print(f"  {key}: {value[:50]}..." if len(value) > 50 else f"  {key}: {value}")

# Multiple ways to get OpenAI API key for Railway compatibility
OPENAI_API_KEY = None
possible_keys = ['OPENAI_API_KEY', 'openai_api_key', 'OPENAI_KEY']

for key in possible_keys:
    value = os.getenv(key) or os.environ.get(key)
    if value:
        OPENAI_API_KEY = value
        print(f"✅ Found OpenAI API key via {key}: {value[:20]}...")
        break

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
    print("Available env vars containing 'OPENAI':", [k for k in os.environ.keys() if 'OPENAI' in k.upper()])
    print("All env var keys:", list(os.environ.keys())[:10])  # Show first 10 keys

def initialize_openai_client():
    """Initialize OpenAI client with multiple fallback methods"""
    if not OPENAI_API_KEY:
        print("⚠️ OpenAI client not initialized - no API key found")
        return None
    
    if not OpenAI:
        print("❌ OpenAI library not available")
        return None
    
    # Clear any proxy-related environment variables that might interfere
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    original_proxy_values = {}
    
    for var in proxy_vars:
        if var in os.environ:
            original_proxy_values[var] = os.environ[var]
            print(f"🔄 Temporarily removing proxy env var: {var}")
            del os.environ[var]
    
    try:
        # Method 1: Clean initialization without any proxy interference
        print("🔄 Method 1: Clean OpenAI client initialization...")
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=60.0,  # Increase timeout for Railway
            max_retries=3
        )
        print("✅ Clean OpenAI client initialization successful")
        return client
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    try:
        # Method 2: Custom HTTP client with Railway-friendly settings
        print("🔄 Method 2: Railway-optimized HTTP client...")
        import httpx
        
        # Create HTTP client with extended timeouts for Railway
        http_client = httpx.Client(
            proxies={},
            timeout=httpx.Timeout(60.0, connect=30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True
        )
        
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client,
            max_retries=5
        )
        print("✅ Railway-optimized client initialization successful")
        return client
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    try:
        # Method 3: Basic client with minimal configuration
        print("🔄 Method 3: Minimal OpenAI client...")
        client = OpenAI(OPENAI_API_KEY)
        print("✅ Minimal client initialization successful")
        return client
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    finally:
        # Restore proxy environment variables
        for var, value in original_proxy_values.items():
            os.environ[var] = value
            print(f"🔄 Restored proxy env var: {var}")
    
    print("❌ All OpenAI client initialization methods failed")
    return None

# Initialize OpenAI client
client = initialize_openai_client()

def test_openai_connection():
    """Test OpenAI connection with simple API call"""
    if not client:
        return False
        
    try:
        print("🧪 Testing OpenAI connection...")
        # Try a simple models list call first
        models = client.models.list()
        print("✅ OpenAI models list successful")
        return True
    except Exception as e:
        print(f"⚠️ OpenAI connection test failed: {e}")
        
        # Try a basic embedding call as alternative test
        try:
            print("🧪 Trying alternative connection test...")
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input="test connection",
                timeout=30.0
            )
            print("✅ OpenAI embedding test successful")
            return True
        except Exception as e2:
            print(f"❌ All connection tests failed: {e2}")
            return False

if client:
    print("✅ OpenAI client ready for use")
    connection_ok = test_openai_connection()
    if connection_ok:
        print("🎯 OpenAI API is fully accessible")
    else:
        print("⚠️ OpenAI API connection issues detected")
else:
    print("❌ OpenAI client not available - RAG functionality will be limited")

# Global variables for RAG system
candidates_data = []
vector_index = {}
embedding_cache = {}

def load_candidate_data():
    """Load candidate data from JSON file with enhanced debugging"""
    global candidates_data
    
    # Show current working directory and available files for debugging
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    try:
        # List files in current directory for debugging
        files = os.listdir('.')
        json_files = [f for f in files if f.endswith('.json')]
        print(f"📂 Available JSON files: {json_files}")
        
        data_path = os.getenv('DATA_FILE_PATH', 'duvari_NFA_data.json')
        print(f"🎯 Attempting to load data from: {data_path}")
        
        if os.path.exists(data_path):
            print(f"✅ Found data file at: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
        else:
            print(f"⚠️ Data file not found at {data_path}, trying relative path...")
            # Fallback to relative path
            with open('duvari_NFA_data.json', 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
            print("✅ Loaded data from relative path")
        
        print(f"✅ Successfully loaded {len(candidates_data)} candidates")
        
        # Show sample of data for verification
        if candidates_data:
            sample = candidates_data[0]
            print(f"📋 Sample candidate: {sample.get('FirstName', 'N/A')} {sample.get('LastName', 'N/A')}")
            print(f"📋 Sample tags: {sample.get('Tags', [])[:3]}...")  # Show first 3 tags
            print(f"📋 Sample location: {sample.get('City', 'N/A')}, {sample.get('State', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading candidate data: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        candidates_data = []
        return False

def candidate_to_text(candidate):
    """Convert candidate object to searchable text"""
    parts = [
        f"{candidate.get('FirstName', '')} {candidate.get('LastName', '')}",
        candidate.get('ExperienceLevel', ''),
        f"{candidate.get('YearsExperience', 0)} years experience",
        ' '.join(candidate.get('Tags', [])),
        f"{candidate.get('City', '')}, {candidate.get('State', '')}",
        candidate.get('Status', ''),
        candidate.get('Availability', '')
    ]
    return '. '.join(filter(None, parts))

def get_embedding(text):
    """Get OpenAI embedding for text with caching and retry logic"""
    if not client:
        print("❌ OpenAI client not available")
        return None
        
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    # Retry logic with exponential backoff
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries} - Generating embedding...")
            
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                timeout=60.0  # Increase timeout
            )
            embedding = response.data[0].embedding
            embedding_cache[text_hash] = embedding
            print(f"🧠 Generated embedding for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            # Check for specific error types
            if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                if attempt < max_retries - 1:  # Don't wait after last attempt
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Network error, waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print("❌ All retry attempts failed due to network issues")
                    return None
            else:
                # For non-network errors, don't retry
                print(f"❌ Non-network error, not retrying: {e}")
                return None
    
    return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    if NUMPY_AVAILABLE:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        # Pure Python implementation
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)

# Removed bulk vector index building - now using on-demand embedding generation

def parse_search_query(query):
    """Parse natural language query using OpenAI"""
    if not client:
        print("❌ OpenAI client not available, using fallback parsing")
        return fallback_parse(query)
        
    system_prompt = """You are a job search query parser. Convert natural language queries into structured parameters for searching a job applicant database.

Return a JSON object with these possible fields:
- skills: array of technical skills mentioned
- location: location/city/state mentioned  
- title: job title or role mentioned
- experience_min: minimum years of experience (if mentioned)
- experience_max: maximum years of experience (if mentioned)
- remote: boolean if remote work is mentioned
- availability: availability timeline
- status: candidate status

Examples:
Query: "Find Python developers in Vermont with 10+ years experience"
Response: {"skills": ["Python"], "location": "Vermont", "experience_min": 10}

Query: "Senior DevOps engineers with AWS and Terraform experience"  
Response: {"skills": ["DevOps", "AWS", "Terraform"], "title": "senior"}

Only include fields that are explicitly mentioned."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        print(f"🧠 Parsed query: {query} → {content}")
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Error parsing query: {e}")
        return fallback_parse(query)

def fallback_parse(query):
    """Basic keyword extraction as fallback when OpenAI is unavailable"""
    query_lower = query.lower()
    params = {}
    
    # Common skills to look for
    skills = ['python', 'java', 'javascript', 'react', 'node.js', 'nodejs', 'aws', 
             'azure', 'devops', 'terraform', 'docker', 'kubernetes', 'sql', 'mongodb',
             'angular', 'vue', 'django', 'flask', 'spring', '.net', 'c#', 'c++',
             'golang', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'linux',
             'machine learning', 'ml', 'ai', 'tensorflow', 'pytorch']
    
    found_skills = []
    for skill in skills:
        if skill in query_lower:
            # Normalize skill names
            if skill == 'nodejs':
                found_skills.append('Node.js')
            elif skill == 'golang':
                found_skills.append('Go')
            elif skill == 'machine learning' or skill == 'ml':
                found_skills.append('Machine Learning')
            else:
                found_skills.append(skill.title())
    
    if found_skills:
        params['skills'] = list(set(found_skills))  # Remove duplicates
    
    # Check for experience levels
    if any(word in query_lower for word in ['senior', 'sr.', 'lead', 'principal']):
        params['title'] = 'senior'
    elif any(word in query_lower for word in ['junior', 'jr.', 'entry']):
        params['title'] = 'junior'
    elif 'mid' in query_lower:
        params['title'] = 'mid-level'
    
    # Check for remote
    if 'remote' in query_lower:
        params['remote'] = True
    
    # Check for years of experience
    import re
    exp_pattern = r'(\d+)\+?\s*years?'
    exp_match = re.search(exp_pattern, query_lower)
    if exp_match:
        params['experience_min'] = int(exp_match.group(1))
    
    # Check for locations
    states = ['vermont', 'california', 'new york', 'texas', 'florida', 'massachusetts']
    for state in states:
        if state in query_lower:
            params['location'] = state.title()
            break
    
    print(f"🔄 Fallback parsed query: {query} → {params}")
    return params

def semantic_search(query, k=10, threshold=0.5, parsed_params=None):
    """Perform on-demand semantic search - only generate embeddings for relevant candidates"""
    print(f"🔍 Performing on-demand semantic search for: '{query}' (threshold: {threshold})")
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("❌ Could not get query embedding")
        return []
    
    # First, filter candidates using keyword criteria to reduce embedding calls
    relevant_candidates = []
    if parsed_params:
        print(f"🎯 Pre-filtering candidates using parsed params: {parsed_params}")
        for candidate in candidates_data:
            if matches_criteria(candidate, parsed_params):
                relevant_candidates.append(candidate)
        print(f"📋 Pre-filtered to {len(relevant_candidates)} relevant candidates")
    
    # If no keyword matches, use a broader set but limit to reasonable size
    if not relevant_candidates:
        print("🔄 No keyword matches, using broader candidate set")
        # Limit to first 50 candidates to control costs
        relevant_candidates = candidates_data[:min(50, len(candidates_data))]
        print(f"📋 Using {len(relevant_candidates)} candidates for semantic search")
    
    # Generate embeddings on-demand for relevant candidates only
    print(f"🧠 Generating embeddings for {len(relevant_candidates)} candidates...")
    similarities = []
    successful_embeddings = 0
    
    for i, candidate in enumerate(relevant_candidates):
        contact_id = candidate.get('ContactId')
        if not contact_id:
            continue
            
        # Check if embedding is already cached
        text = candidate_to_text(candidate)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        candidate_embedding = None
        if text_hash in embedding_cache:
            candidate_embedding = embedding_cache[text_hash]
        else:
            # Generate embedding on-demand
            candidate_embedding = get_embedding(text)
            if candidate_embedding:
                successful_embeddings += 1
        
        if candidate_embedding:
            similarity = cosine_similarity(query_embedding, candidate_embedding)
            
            if similarity >= threshold:
                similarities.append({
                    'candidate': candidate,
                    'similarity': similarity,
                    'relevance_score': int(similarity * 100),
                    'text': text
                })
        
        # Show progress for larger sets
        if len(relevant_candidates) > 10 and (i + 1) % 10 == 0:
            print(f"📊 Progress: {i+1}/{len(relevant_candidates)} processed, {successful_embeddings} embeddings generated")
    
    print(f"💰 Generated {successful_embeddings} new embeddings for this search")
    
    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    results = similarities[:k]
    
    print(f"✅ Found {len(results)} semantic matches above threshold {threshold}")
    
    # If no results with threshold, try with very low threshold
    if not results and similarities:
        print(f"🔄 No results with threshold {threshold}, trying with lower threshold...")
        low_threshold = max(0.3, threshold * 0.7)
        
        for item in similarities:
            if item['similarity'] >= low_threshold:
                results.append(item)
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:k]
        print(f"✅ Found {len(results)} matches with lower threshold {low_threshold:.3f}")
    
    return results

def keyword_search(query, params):
    """Traditional keyword-based search"""
    results = []
    
    for candidate in candidates_data:
        if matches_criteria(candidate, params):
            results.append({
                'candidate': candidate,
                'similarity': 0.8,  # Default similarity for keyword matches
                'relevance_score': 80
            })
    
    return results[:20]  # Limit results

def matches_criteria(candidate, params):
    """Check if candidate matches search criteria"""
    # Check skills
    if params.get('skills'):
        candidate_tags = [tag.lower() for tag in candidate.get('Tags', [])]
        required_skills = [skill.lower() for skill in params['skills']]
        
        if not any(skill in ' '.join(candidate_tags) for skill in required_skills):
            return False
    
    # Check location
    if params.get('location'):
        location = params['location'].lower()
        candidate_city = candidate.get('City', '').lower()
        candidate_state = candidate.get('State', '').lower()
        
        if location not in candidate_city and location not in candidate_state:
            return False
    
    # Check experience
    if params.get('experience_min'):
        if candidate.get('YearsExperience', 0) < params['experience_min']:
            return False
    
    # Check title/level
    if params.get('title'):
        title = params['title'].lower()
        exp_level = candidate.get('ExperienceLevel', '').lower()
        
        title_mapping = {
            'senior': ['senior', 'principal', 'staff'],
            'mid': ['mid-level', 'mid'],
            'junior': ['junior', 'entry'],
            'principal': ['principal', 'staff']
        }
        
        if title in title_mapping:
            if not any(level in exp_level for level in title_mapping[title]):
                return False
    
    return True

def format_results(candidates, query):
    """Format search results using OpenAI"""
    if not candidates:
        return "I couldn't find any candidates matching your criteria. Try adjusting your search terms."

    # If OpenAI client is not available, use basic formatting
    if not client:
        count = len(candidates)
        return f"Found {count} candidate{'s' if count != 1 else ''} matching '{query}'. Results include candidates with relevant skills and experience levels."

    system_prompt = """You are a helpful recruitment assistant. Format the candidate search results in a conversational, professional manner.

Include:
- Brief summary of how many candidates were found
- Highlight top 3-5 candidates with key details  
- Mention relevant skills, experience, and location
- Keep it concise but informative
- Use a friendly, professional tone"""

    candidates_text = json.dumps([c.get('candidate', c) for c in candidates[:5]], indent=2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: {query}\n\nCandidates found:\n{candidates_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        formatted_response = response.choices[0].message.content.strip()
        print(f"🎨 Formatted results for query: {query}")
        return formatted_response
        
    except Exception as e:
        print(f"❌ Error formatting results: {e}")
        return f"Found {len(candidates)} candidate{'s' if len(candidates) != 1 else ''} matching your criteria."

# Routes
@app.route('/')
def index():
    """Main search interface"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with full RAG functionality"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'hybrid')  # semantic, keyword, hybrid
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"🔍 Processing search: '{query}' (type: {search_type})")
        
        # Parse query with OpenAI
        parsed_params = parse_search_query(query)
        
        # Perform search based on type with more lenient thresholds
        if search_type == 'semantic':
            results = semantic_search(query, k=10, threshold=0.4, parsed_params=parsed_params)
        elif search_type == 'keyword':
            results = keyword_search(query, parsed_params)
        else:  # hybrid
            semantic_results = semantic_search(query, k=8, threshold=0.4, parsed_params=parsed_params)
            keyword_results = keyword_search(query, parsed_params)
            
            # Merge results (semantic gets priority)
            seen_ids = set()
            results = []
            
            for result in semantic_results + keyword_results:
                candidate = result.get('candidate', result)
                contact_id = candidate.get('ContactId')
                if contact_id not in seen_ids:
                    results.append(result)
                    seen_ids.add(contact_id)
                    
                if len(results) >= 10:  # Limit to top 10
                    break
        
        # Format results with OpenAI
        formatted_response = format_results(results, query)
        
        # Extract just candidates for response
        candidates = [r.get('candidate', r) for r in results]
        similarities = [r.get('similarity', 0) for r in results]
        relevance_scores = [r.get('relevance_score', 0) for r in results]
        
        response_data = {
            'query': query,
            'search_type': search_type,
            'parsed_params': parsed_params,
            'candidates': candidates,
            'similarities': similarities,
            'relevance_scores': relevance_scores,
            'formatted_response': formatted_response,
            'count': len(candidates),
            'metadata': {
                'vector_index_size': len(vector_index),
                'embedding_cache_size': len(embedding_cache),
                'search_timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"✅ Search completed: {len(candidates)} candidates found")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag-search', methods=['POST'])
def rag_search():
    """Advanced RAG search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 10)
        threshold = data.get('threshold', 0.7)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"🧠 RAG Search: '{query}' (k={k}, threshold={threshold})")
        
        start_time = time.time()
        results = semantic_search(query, k=k, threshold=threshold)
        search_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'search_time_ms': round(search_time, 2),
            'threshold': threshold,
            'vector_stats': {
                'total_vectors': len(vector_index),
                'cache_size': len(embedding_cache)
            }
        })
        
    except Exception as e:
        print(f"❌ RAG search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_candidates': len(candidates_data),
        'embedding_cache_size': len(embedding_cache),
        'openai_configured': bool(OPENAI_API_KEY),
        'openai_client_ready': client is not None,
        'system_mode': 'on-demand embedding generation',
        'cost_optimization': 'embeddings generated only during searches',
        'system_status': 'operational'
    })

@app.route('/health')
def health():
    """Health check endpoint with detailed debugging"""
    return jsonify({
        'status': 'healthy',
        'candidates_loaded': len(candidates_data),
        'embedding_cache_size': len(embedding_cache),
        'openai_configured': bool(OPENAI_API_KEY),
        'openai_client_ready': client is not None,
        'system_mode': 'on-demand embeddings',
        'cost_optimized': True,
        'total_env_vars': len(os.environ),
        'port': os.getenv('PORT', 'not set'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear embedding cache"""
    try:
        global embedding_cache
        cache_size = len(embedding_cache)
        embedding_cache = {}
        return jsonify({
            'status': 'success',
            'message': f'Cleared {cache_size} cached embeddings',
            'cache_size': 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize system on startup (runs for both gunicorn and direct execution)
print("🚀 Starting Duvari RAG Candidate Search System...")

# Load data
if load_candidate_data():
    print("✅ Candidate data loaded")
else:
    print("❌ Failed to load candidate data")

# Skip bulk vector index building - generate embeddings on-demand during searches
print("🎯 Vector embeddings will be generated on-demand during searches")
print(f"🎯 System ready with {len(candidates_data)} candidates loaded")
print(f"💰 Cost-optimized: Only generating embeddings for searched candidates")

# Only run Flask directly if called as main script
if __name__ == '__main__':
    print("🌐 Starting Flask server directly...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)