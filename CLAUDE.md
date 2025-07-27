# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NetOps ChatBot API - Enterprise-grade RAG-enhanced Universal AI-driven ChatOps platform with Smart Command Suggestions. The system implements a revolutionary three-tier discovery system with progressive performance optimization: Redis Cache (1-5ms) → PostgreSQL Database (10-20ms) → LLM Discovery (1-2s), enhanced with RAG-powered command validation using 40k+ vendor documentation chunks.

**Current Phase:** Phase 6 RAG Integration Complete - Smart Command Suggestions Operational

## Common Development Commands

### Running the Application
```bash
# Start the FastAPI application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python app/main.py
```

### Code Quality Tools
```bash
# Format code with Black (88-character line width)
black .

# Sort imports
isort .

# Lint with Ruff
ruff check .

# Type checking
mypy app/
```

### Testing
```bash
# Run tests
pytest

# Run async tests
pytest --asyncio-mode=auto

# Run RAG integration tests
python test_rag_integration.py

# Run specific test file
pytest test_rag_integration.py
```

### Docker Services
```bash
# Start Redis cache service
cd cache && docker-compose up -d

# Start PostgreSQL database service
cd database && docker-compose up -d

# Create shared network (first time only)
docker network create netops-network
```

### Database Operations
```bash
# Run database migrations
alembic upgrade head

# Seed database with initial data
python database/seed_postgres.py
```

## High-Level Architecture

### Enhanced Request Flow with RAG Validation
```
User Input → LLM Intent Parser → Three-Tier Discovery →
                                    ↓
[NEW] RAG Command Validation → Command Execution → Universal Formatter → Response
```

### Three-Tier Discovery System
The system implements a progressive performance optimization approach:

1. **Tier 1: Redis Cache** (1-5ms)
   - Stores frequently used command mappings
   - Immediate response for known commands
   - Located in `cache/` directory

2. **Tier 2: PostgreSQL Database** (10-20ms)
   - Persistent storage for learned commands
   - Vendor-specific command mapping
   - Seeded with 1,026 community commands
   - Located in `database/` directory

3. **Tier 3: LLM Discovery** (1-2s)
   - AI-powered command discovery for unknown commands
   - Uses LM Studio with LLama 3.1 8B on RTX 5090
   - Results are cached for future use

### RAG Command Validation (NEW - Phase 6)
- **Service Integration:** Connects to Barrow RAG service (http://192.168.1.11:8001)
- **Documentation Coverage:** 40k+ vendor documentation chunks
- **Validation Speed:** 1-3s per command (cached for future speed)
- **Smart Suggestions:** Invalid commands trigger documentation-backed suggestions
- **Interactive Workflow:** User confirmation for command corrections
- **Graceful Fallback:** System continues when RAG service unavailable

### Core Components

1. **Universal Request Processor** (`app/services/universal_request_processor.py`)
   - Central processing unit for all requests
   - Implements three-tier discovery logic with RAG validation
   - Routes to appropriate services based on discovery results
   - Enhanced with smart command suggestion workflow

2. **Pure LLM Intent Parser** (`app/services/pure_llm_intent_parser.py`)
   - Uses pure LLM understanding - NO pattern matching
   - Extracts device, action, and parameters from natural language
   - Infinite scalability with any command format

3. **RAG Command Validation** (`app/services/rag_command_validation.py`)
   - Validates discovered commands using 40k+ documentation chunks
   - Provides smart suggestions with confidence scores
   - Documentation-backed corrections with explanations
   - Graceful fallback when Barrow service unavailable

4. **Hybrid Command Discovery** (`app/services/hybrid_command_discovery.py`)
   - LLM-powered three-tier discovery service
   - Progressive optimization: Redis → PostgreSQL → LLM
   - Learns from every interaction for future speed

5. **Universal Formatter** (`app/services/universal_formatter.py`)
   - Enterprise anti-hallucination formatting
   - Expert network engineer response persona
   - 100% ground truth - zero data fabrication
   - Content grounding validation

6. **Network Executor** (`app/services/executor.py`)
   - Executes validated commands on network devices
   - Supports multiple vendors (Cisco, Arista, Juniper)
   - Uses Nornir for device management

7. **Local LLM Normalizer** (`app/services/local_llm_normalizer.py`)
   - Direct LM Studio integration
   - Intent normalization without hardcoded patterns
   - Seamless fallback when LLM unavailable

### API Structure

- **Main Application**: `app/main.py` - FastAPI application with lifespan management
- **Routers**:
  - `universal.py` - Main endpoint for all network operations
  - `analytics.py` - Performance monitoring and statistics
  - `discovery.py` - Command discovery management
- **Models**: Data models for requests, responses, and database schemas
- **Services**: Business logic and integrations

### Network Lab Environment

The `labs/` directory contains Containerlab configurations for testing:
- Arista switches (leaf1, leaf2, spine1)
- Pre-configured with startup configs
- Accessible via SSH for testing network commands

### Key Dependencies

- **FastAPI**: Web framework
- **Nornir**: Network automation framework
- **SQLAlchemy**: Database ORM
- **Redis**: Caching layer
- **Ollama/OpenAI**: LLM integration
- **Pydantic**: Data validation

### Environment Variables

Key environment variables to configure:

**Database & Cache:**
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `REDIS_HOST`, `REDIS_PORT`

**LLM Configuration:**
- `LM_STUDIO_BASE_URL=http://192.168.1.11:1234` (LM Studio API)
- `OPENAI_API_KEY` (if using OpenAI as fallback)
- `OLLAMA_BASE_URL` (if using local Ollama)

**RAG Integration (NEW):**
- `RAG_SERVICE_ENABLED=true`
- `RAG_SERVICE_URL=http://192.168.1.11:8001` (Barrow service)
- `RAG_SERVICE_TIMEOUT=10`
- `RAG_VALIDATION_CONFIDENCE_THRESHOLD=0.8`
- `RAG_SUGGESTION_CONFIDENCE_THRESHOLD=0.7`
- `RAG_AUTO_APPLY_THRESHOLD=0.95`

**Mattermost Integration:**
- `MATTERMOST_URL=http://192.168.1.100:8065`
- `MATTERMOST_TOKEN` (bot token)
- `MATTERMOST_TEAM_NAME=netops`

### Performance Evolution

The system learns and optimizes over time:
- Week 1: ~80% LLM usage (learning phase)
- Month 6: ~70% Redis + 20% PostgreSQL + 10% LLM
- Month 12: ~80% Redis + 15% PostgreSQL + 5% LLM

This progressive optimization reduces response times and API costs while maintaining flexibility for new commands.

## Key Features

### Enterprise Anti-Hallucination System
- **Zero-tolerance data fabrication** protection
- **Content grounding validation** - works across ALL vendors/commands
- **Expert network engineer response** persona - ground truth only
- **Safe fallbacks** when validation fails

### RAG-Enhanced Command Validation
- **Smart suggestions** for invalid commands with documentation backing
- **Interactive correction workflow** with user confirmation
- **40k+ vendor documentation chunks** for accurate validation
- **Learning integration** - corrections stored for future improvements

### Pure LLM Understanding
- **NO pattern matching** - infinite scalability
- **Natural language processing** for any command format
- **Multi-vendor support** without hardcoded rules
- **Seamless conversation flow** with context understanding

## API Endpoints

### Main Endpoints
- `/api/v1/universal` - Universal endpoint for all network operations (RAG enhanced)
- `/api/v1/chat` - Chat interface simulation
- `/api/v1/query` - Quick URL-based queries
- `/api/v1/analytics/dashboard` - Three-tier performance dashboard
- `/api/v1/analytics/performance/current` - Real-time performance metrics
- `/docs` - Interactive API documentation

### Health & Status
- `/` - System information with three-tier discovery status
- `/health` - Detailed health check with service statuses
- `/api/v1/discovery/stats` - Three-tier discovery statistics
- `/api/v1/llm/status` - LLM service status

## Testing Smart Command Suggestions

### Test Invalid Command (Should Get Suggestions)
```bash
curl -X POST http://localhost:8000/api/v1/universal \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "show interface states on spine1",
    "interface_type": "api",
    "output_format": "json",
    "enable_command_validation": true
  }'
```

### Test Valid Command (Should Execute Normally)
```bash
curl -X POST http://localhost:8000/api/v1/universal \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "show version on spine1",
    "interface_type": "api",
    "output_format": "json",
    "enable_command_validation": true
  }'
```

## Code Quality Standards

- **Formatter:** Black with 88-character line width
- **Linter:** Ruff with comprehensive rule set
- **Type Checking:** mypy for static type analysis
- **Import Sorting:** isort for consistent imports
- **Exception Handling:** Proper exception chaining required
- **VS Code Integration:** Full support for development
