# Student Health AI Assistant

A comprehensive AI assistant for student health and wellness queries that combines multiple knowledge sources to provide evidence-based health advice.

## Overview

This project consists of two main components:
1. **`app.py`** - A multi-tool AI agent that intelligently selects the best information source
2. **`evaluation_rag.ipynb`** - A Jupyter notebook for evaluating different RAG (Retrieval-Augmented Generation) approaches

## Features

### Multi-Source Information Retrieval
- **Local Health Database**: RAG-based search through student health PDF documents
- **Web Search**: Current health information and trends via Tavily
- **Academic Research**: Scholarly articles and papers via arXiv
- **Medical Literature**: PubMed database for clinical studies and medical research

### Intelligent Tool Selection
The AI agent automatically determines which information source is most appropriate for each query:
- Student health topics → Local database
- Current trends/news → Web search
- Academic research → arXiv papers
- Medical evidence → PubMed studies

## File Structure

```
11_Certification_Challenge/
├── app.py                          # Main AI assistant application
├── evaluation_rag.ipynb            # RAG evaluation notebook
├── data/                           # Student health PDF documents
│   ├── Accessible-online-version-Manage-Stress-Workbook-PennState-final-2i5sewu.pdf
│   ├── CollegeStudentMentalHealthActionToolkit.pdf
│   ├── guide-to-college-student-nutrition-survival-flccc.pdf
│   ├── SHS_PatientEd_SleepHygiene.pdf
│   ├── Sleep_Healthy Habits.pdf
│   ├── Sleep-LetsCU.pdf
│   └── Workouts for College Students.pdf
├── frontend/                       # Next.js web interface
├── server.py                       # Flask server for API endpoints
└── pyproject.toml                 # Python dependencies
```

## Detailed Component Analysis

### app.py - Main AI Assistant

#### Core Components:

**1. Document Processing**
- Loads PDF documents from `data/` directory using PyMuPDF
- Splits documents into 750-token chunks for optimal retrieval
- Uses tiktoken for accurate token counting

**2. Vector Store Setup**
- Creates embeddings using OpenAI's text-embedding-3-small model
- Stores vectors in Qdrant (in-memory for demo)
- Enables semantic search through health documents

**3. RAG Model**
- Custom prompt template for health-specific responses
- Uses GPT-4o-mini for response generation
- Context-aware answers based on retrieved documents

**4. Tool Definitions**
- `rag_search()`: Searches local health database
- `web_search()`: Searches current web information via Tavily
- `research_papers()`: Searches academic papers via arXiv
- `medical_research()`: Searches medical literature via PubMed

**5. LangGraph Workflow**
- Multi-agent system that intelligently routes queries
- Determines when to use tools vs. provide direct answers
- Handles conversation state and tool execution

#### Key Functions:
```python
# Main entry point for queries
agent_chain_with_formatting = convert_inputs | simple_agent_graph | parse_output
```

### evaluation_rag.ipynb - RAG Evaluation

#### Purpose:
Evaluates different RAG approaches to determine the most effective retrieval method for student health queries.

#### Evaluated Approaches:

**1. Naive Retrieval Chain**
- Basic RAG implementation
- Simple document retrieval + LLM response generation
- Baseline for comparison

**2. Contextual Compression with Cohere Rerank**
- Uses Cohere's rerank-v3.5 model
- Re-ranks retrieved documents for better relevance
- Improves answer quality through better document selection

**3. Multi-Query Retriever**
- Generates multiple queries from user input
- Searches with different query variations
- Combines results for comprehensive coverage

**4. Parent Document Retriever**
- Hierarchical document structure
- Retrieves parent documents and relevant child chunks
- Maintains document context and relationships

**5. Ensemble Retriever**
- Combines all retrieval methods with equal weighting
- Aggregates results from multiple approaches
- Provides most comprehensive information coverage

#### Key Metrics Evaluated:
- Answer relevance and accuracy
- Information comprehensiveness
- Response quality for different query types
- Performance comparison across methods

## Setup Instructions

### Prerequisites
- Python 3.9+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd 11_Certification_Challenge
```

2. **Install dependencies**
```bash
uv sync
```

3. **Set up API keys**
```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
export LANGCHAIN_API_KEY="your-langsmith-key"
```

### Running the Application

**Option 1: Direct Python execution**
```bash
python app.py
```

**Option 2: Using the Flask server**
```bash
python server.py
```

**Option 3: Jupyter notebook evaluation**
```bash
jupyter notebook evaluation_rag.ipynb
```

## Usage Examples

### Health Query Examples:
- "What snacks should I stock in my dorm room fridge?"
- "How can I manage stress during exams?"
- "What are good sleep habits for college students?"
- "What exercises can I do in my dorm room?"
- "How much sleep do college students need?"

### Tool Selection Logic:
- **Nutrition/exercise advice** → Local database + Web search
- **Stress management** → Local database + Medical research
- **Sleep optimization** → Local database + Academic research
- **Current health trends** → Web search
- **Medical conditions** → PubMed research

## Technical Architecture

### Data Flow:
1. User query received
2. Agent analyzes query intent
3. Appropriate tools selected
4. Information retrieved from multiple sources
5. Responses synthesized and formatted
6. Final answer delivered to user

### Key Technologies:
- **LangChain**: RAG framework and tool integration
- **LangGraph**: Multi-agent workflow orchestration
- **OpenAI**: LLM and embedding models
- **Qdrant**: Vector database for document storage
- **Tavily**: Web search API
- **arXiv**: Academic paper search
- **PubMed**: Medical literature search

## Performance Considerations

### Optimization Features:
- Token-based text splitting for optimal chunk sizes
- In-memory vector storage for fast retrieval
- Tool selection based on query type
- Response caching for repeated queries

### Scalability:
- Modular tool architecture allows easy addition of new sources
- Configurable chunk sizes and overlap
- Support for persistent vector storage
- Extensible agent workflow

## Troubleshooting

### Common Issues:

**1. Import Errors**
- Ensure all dependencies are installed: `uv sync`
- Check Python version compatibility

**2. API Key Errors**
- Verify all required API keys are set as environment variables
- Test API key validity with simple requests

**3. Document Loading Issues**
- Ensure PDF files are in the `data/` directory
- Check file permissions and format compatibility

**4. Vector Store Issues**
- Clear and rebuild vector store if corrupted
- Check embedding model availability

## Contributing

### Adding New Tools:
1. Define new tool function with `@tool` decorator
2. Add to `tool_belt` list in `app.py`
3. Update agent workflow if needed

### Adding New Documents:
1. Place PDF files in `data/` directory
2. Restart application to rebuild vector store
3. Test retrieval with relevant queries

## License

This project is part of the AIE7 Certification Challenge and follows the associated licensing terms. 