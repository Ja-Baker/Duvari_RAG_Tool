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
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ Clean OpenAI client initialization successful")
        return client
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    try:
        # Method 2: Force no proxy configuration
        print("🔄 Method 2: OpenAI client with explicit no-proxy configuration...")
        import httpx
        
        # Create a custom HTTP client with no proxy
        http_client = httpx.Client(proxies={})
        
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client
        )
        print("✅ No-proxy OpenAI client initialization successful")
        return client
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    try:
        # Method 3: Use requests-based approach
        print("🔄 Method 3: Alternative HTTP client initialization...")
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=30.0,
            max_retries=2
        )
        print("✅ Alternative HTTP client initialization successful")
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

if client:
    print("✅ OpenAI client ready for use")
    # Quick test
    try:
        models = client.models.list()
        print("✅ OpenAI client API test successful")
    except Exception as test_error:
        print(f"⚠️ OpenAI client API test failed: {test_error}")
else:
    print("❌ OpenAI client not available - RAG functionality will be limited")

# Global variables for RAG system
candidates_data = []
vector_index = {}
embedding_cache = {}

def load_candidate_data():
    """Load candidate data from JSON file"""
    global candidates_data
    try:
        data_path = os.getenv('DATA_FILE_PATH', 'duvari_NFA_data.json')
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
        else:
            # Fallback to relative path
            with open('duvari_NFA_data.json', 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
        
        print(f"✅ Loaded {len(candidates_data)} candidates")
        return True
    except Exception as e:
        print(f"❌ Error loading candidate data: {e}")
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
    """Get OpenAI embedding for text with caching"""
    if not client:
        print("❌ OpenAI client not available")
        return None
        
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response.data[0].embedding
        embedding_cache[text_hash] = embedding
        print(f"🧠 Generated embedding for: {text[:50]}...")
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
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

def build_vector_index():
    """Build vector index for all candidates"""
    global vector_index
    print("🔧 Building vector index...")
    
    for candidate in candidates_data:
        contact_id = candidate.get('ContactId')
        if contact_id:
            text = candidate_to_text(candidate)
            embedding = get_embedding(text)
            if embedding:
                vector_index[contact_id] = {
                    'embedding': embedding,
                    'candidate': candidate,
                    'text': text
                }
    
    print(f"✅ Built vector index with {len(vector_index)} candidates")

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

def semantic_search(query, k=10, threshold=0.7):
    """Perform semantic search using vector similarity"""
    print(f"🔍 Performing semantic search for: '{query}'")
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    # Calculate similarities
    similarities = []
    for contact_id, data in vector_index.items():
        similarity = cosine_similarity(query_embedding, data['embedding'])
        if similarity >= threshold:
            similarities.append({
                'candidate': data['candidate'],
                'similarity': similarity,
                'relevance_score': int(similarity * 100),
                'text': data['text']
            })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    results = similarities[:k]
    
    print(f"✅ Found {len(results)} semantic matches")
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
        
        # Perform search based on type
        if search_type == 'semantic':
            results = semantic_search(query, k=10, threshold=0.6)
        elif search_type == 'keyword':
            results = keyword_search(query, parsed_params)
        else:  # hybrid
            semantic_results = semantic_search(query, k=8, threshold=0.6)
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
        'vector_index_size': len(vector_index),
        'embedding_cache_size': len(embedding_cache),
        'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
        'system_status': 'operational'
    })

@app.route('/health')
def health():
    """Health check endpoint with detailed debugging"""
    return jsonify({
        'status': 'healthy',
        'candidates_loaded': len(candidates_data),
        'vector_index_ready': len(vector_index) > 0,
        'openai_configured': bool(OPENAI_API_KEY),
        'openai_client_ready': client is not None,
        'total_env_vars': len(os.environ),
        'openai_env_vars': [k for k in os.environ.keys() if 'OPENAI' in k.upper()],
        'port': os.getenv('PORT', 'not set'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """Rebuild the vector index"""
    try:
        build_vector_index()
        return jsonify({
            'status': 'success',
            'vector_count': len(vector_index),
            'message': 'Vector index rebuilt successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize system on startup
if __name__ == '__main__':
    print("🚀 Starting Duvari RAG Candidate Search System...")
    
    # Load data
    if load_candidate_data():
        print("✅ Candidate data loaded")
    else:
        print("❌ Failed to load candidate data")
    
    # Build vector index
    if candidates_data:
        build_vector_index()
        print("✅ Vector index built")
    else:
        print("⚠️  No candidates to index")
    
    print(f"🎯 System ready with {len(candidates_data)} candidates and {len(vector_index)} vectors")
    print("🌐 Starting Flask server...")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)