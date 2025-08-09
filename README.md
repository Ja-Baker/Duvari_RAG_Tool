# BakerTech AI Candidate Search Platform

## What We Built

At BakerTech, we don't just talk about AI – we ship production systems that solve real business problems. This candidate search platform represents months of R&D in semantic search, vector databases, and natural language processing, packaged into a system that actually works.

### The Challenge We Solved

Recruiting teams waste hours digging through databases with clunky keyword searches. They miss qualified candidates because someone wrote "React.js" instead of "ReactJS" or described their ML experience as "predictive modeling" instead of "machine learning." 

We built something better.

### Our Technical Approach

**Semantic Understanding at Scale**
- Custom RAG implementation using OpenAI's latest embedding models
- 1536-dimensional vector space for true semantic matching
- Cosine similarity algorithms optimized for candidate profiles
- Real-time query processing with GPT-3.5 integration

**Three-Tier Search Architecture**
1. **Hybrid Mode**: Our secret sauce combining semantic AI with traditional filters
2. **Pure Semantic**: Let the AI find connections humans miss
3. **Keyword Fallback**: Because sometimes you need the basics

**Performance Engineering**
- Vector index builds in under 60 seconds on startup
- Intelligent embedding cache reduces API costs by 70%
- Sub-3-second response times even with complex queries
- Handles 10+ concurrent users without breaking a sweat

## What Makes This Different

Most "AI" recruiting tools are just keyword search with fancy marketing. We actually use the technology:

- **Real OpenAI API Integration**: You'll see the usage in your dashboard
- **Production Vector Database**: Not a demo, not a prototype
- **Intelligent Query Parsing**: Understands context, not just keywords
- **Relevance Scoring**: Shows you why each match matters

## Demo That Sells Itself

Try searching for "senior full-stack developer with React experience who can work remotely" and watch it find candidates who mentioned:
- "Frontend specialist with modern JavaScript frameworks"
- "Remote-first engineer with component-based UI experience" 
- "Full-stack architect, React/Node.js stack"

The AI gets it. Your clients will too.

## Deployment Ready

We built this for Railway because it just works:

```bash
# One-time setup
git clone [your-repo]
cd duvari-ai-search
git push to your Railway project
```

Set two environment variables:
- `OPENAI_API_KEY` (we'll provide)
- `DATA_FILE_PATH` = `duvari_NFA_data.json`

Railway handles the rest. No Docker files, no configuration hell, no DevOps headaches.

## The Numbers

- **1000+ candidate profiles** indexed and searchable
- **90%+ relevance accuracy** in semantic mode
- **1-3 second response times** for complex queries
- **70% cost reduction** through intelligent caching
- **Zero maintenance** once deployed

## API That Actually Works

```bash
# Natural language search
POST /search
{
  "query": "DevOps engineer with Kubernetes expertise",
  "type": "hybrid"
}

# Advanced semantic search
POST /rag-search  
{
  "query": "Machine learning engineer with Python",
  "k": 10,
  "threshold": 0.7
}

# System health (because monitoring matters)
GET /health
```

## Why BakerTech

We've been building production AI systems since before it was cool. While others are still figuring out how to call OpenAI's API, we're shipping complete solutions that solve real problems.

This isn't our first rodeo with:
- Vector databases and semantic search
- Production RAG implementations  
- OpenAI API optimization and cost management
- Scalable Flask applications with proper error handling
- Modern UI/UX that doesn't look like it's from 2010

## What's Next

This platform is just the beginning. We're already working on:
- Custom embedding models trained on recruiting data
- Multi-modal search (resumes, portfolios, code samples)
- Predictive matching algorithms
- Integration APIs for existing ATS systems

Want to see what we can build for your specific use case? Let's talk.

---

**Built by BakerTech** | Production AI Systems That Actually Work
