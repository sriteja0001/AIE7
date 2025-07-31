import os
import getpass
from uuid import uuid4
# RAG Setup
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from Bio import Entrez

from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode

from langgraph.graph import StateGraph, END

from langchain_core.messages import HumanMessage


os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE7 - LangGraph - Certification Challenge - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key: ")


# Load and process documents
directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
student_health_resources = directory_loader.load()

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=0,
    length_function=tiktoken_len,
)

student_health_chunks = text_splitter.split_documents(student_health_resources)

# Setup vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_vectorstore = Qdrant.from_documents(
    documents=student_health_chunks,
    embedding=embedding_model,
    location=":memory:"
)
qdrant_retriever = qdrant_vectorstore.as_retriever()

# RAG prompt and model
HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provided context to answer the user query about student health, wellness, nutrition, stress management, 
sleep, exercise, mental health, or any student success topics. Only use the provided context to answer the query. 
If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("human", HUMAN_TEMPLATE)
])

rag_model = ChatOpenAI(model="gpt-4.1-nano")

# RAG Tool
@tool
def rag_search(query: str) -> str:
    """
    Search the student health knowledge database for information about nutrition, stress management, 
    sleep optimization, exercise, mental health, and overall student wellness. Use this tool when 
    the user asks about student health topics, wellness advice, nutrition guidance, stress relief, 
    sleep improvement, exercise routines, mental health support, or any student success and wellness topics.
    
    IMPORTANT: Provide comprehensive answers directly. Do not ask clarifying questions.
    Give the most relevant and helpful information based on the query.
    """
    # Retrieve relevant documents
    retrieved_docs = qdrant_retriever.invoke(query)
    
    # Generate response
    generator_chain = chat_prompt | rag_model | StrOutputParser()
    response = generator_chain.invoke({"query": query, "context": retrieved_docs})
    
    return response

# Web Search Tool
@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, news, and general knowledge about student health, 
    wellness trends, nutrition research, mental health developments, and general health information. 
    Use this tool when the user asks about recent health news, current wellness trends, 
    general health information not in the student health database, or when they need up-to-date 
    information about health and wellness topics.
    
    IMPORTANT: Provide comprehensive answers directly. Do not ask clarifying questions.
    Give the most relevant and helpful information based on the query.
    """
    tavily_tool = TavilySearchResults(max_results=5)
    return tavily_tool.invoke(query)

# Research Papers Tool
@tool
def research_papers(query: str) -> str:
    """
    Search for academic research papers and scholarly articles about health, nutrition, psychology, 
    exercise science, sleep research, stress management, and wellness studies. Use this tool when 
    the user asks about research papers on health topics, academic studies on wellness, 
    scientific publications about nutrition, exercise, mental health, or when they need scholarly 
    information about health and wellness research.
    
    IMPORTANT: Provide comprehensive answers directly. Do not ask clarifying questions.
    Give the most relevant and helpful information based on the query.
    """
    arxiv_tool = ArxivQueryRun()
    return arxiv_tool.invoke(query)

# Medical Research Tool
@tool
def medical_research(query: str) -> str:
    """
    Search PubMed database for medical research papers, clinical studies, and healthcare publications 
    specifically related to student health, young adult wellness, nutrition science, mental health 
    interventions, sleep medicine, stress physiology, exercise medicine, and preventive healthcare 
    for students. Use this tool when the user asks about medical research on student health topics, 
    clinical trials for wellness interventions, healthcare studies for young adults, medical conditions 
    affecting students, treatments for stress, anxiety, sleep disorders, or any biomedical/healthcare 
    research relevant to student wellness and success.

    Make sure to use this evidence or information you find to actually answer the question that the user asked. you can provide the
    evidence and list another section called "Insight" so the user knows that this is your answer based on what the paper says.
    
    IMPORTANT: Provide comprehensive, evidence-based answers directly. Do not ask clarifying questions.
    Give the most relevant and helpful information based on the query. Include key findings from studies,
    cite sources, and provide actionable insights when possible.
    """
    # Set email for PubMed (required by NCBI)
    Entrez.email = "kitsink@gmail.com"
    
    try:
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=5, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            return f"No medical research papers found for: {query}"
        
        # Get detailed information for the papers
        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="abstract", retmode="text")
        papers_data = handle.read()
        handle.close()
        
        # Parse the papers data to extract titles and abstracts
        papers = []
        current_paper = {}
        lines = papers_data.split('\n')
        
        for line in lines:
            if line.startswith('PMID- '):
                if current_paper:
                    papers.append(current_paper)
                current_paper = {'pmid': line[6:].strip()}
            elif line.startswith('TI  - '):
                current_paper['title'] = line[6:].strip()
            elif line.startswith('AB  - '):
                current_paper['abstract'] = line[6:].strip()
        
        if current_paper:
            papers.append(current_paper)
        
        # Format the response with actual evidence
        if papers:
            response = f"Based on medical research for '{query}', here are the key findings:\n\n"
            
            for i, paper in enumerate(papers[:3], 1):
                response += f"{i}. **{paper.get('title', 'No title available')}**\n"
                response += f"   PMID: {paper.get('pmid', 'N/A')}\n"
                if paper.get('abstract'):
                    # Truncate long abstracts
                    abstract = paper['abstract'][:300] + "..." if len(paper['abstract']) > 300 else paper['abstract']
                    response += f"   Key findings: {abstract}\n"
                response += "\n"
            
            response += "These studies provide evidence-based information on your query. Always consult healthcare professionals for medical advice."
            return response
        else:
            return f"Found {len(record['IdList'])} papers but couldn't retrieve detailed information for: {query}"
        
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"

## tool belt

tool_belt = [
    web_search,
    research_papers,
    medical_research,
    rag_search,
]



model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

model = model.bind_tools(tool_belt)


class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages" : [response]}


uncompiled_graph = StateGraph(AgentState)

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", ToolNode(tool_belt))

uncompiled_graph.set_entry_point("agent")

def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

uncompiled_graph.add_edge("action", "agent")

simple_agent_graph = uncompiled_graph.compile()


def convert_inputs(input_object):
  return {"messages" : [HumanMessage(content=input_object["question"])]}

def parse_output(input_state):
  return input_state["messages"][-1].content

agent_chain_with_formatting = convert_inputs | simple_agent_graph | parse_output

# Test the agent chain with different types of queries
print("Testing agent chain with different tools:")

# Test RAG tool
print("\n1. Testing RAG tool (student health database) PLUS medical research tool")
result1 = agent_chain_with_formatting.invoke({"question": "What are the best nutrition tips for students? Also, find me proven evidence that sleep is good for academic performance in a few sentences."})
print(result1)

# Test Tavily tool
print("\n2. Testing Tavily tool (web search):")
result2 = agent_chain_with_formatting.invoke({"question": "What is the latest research about stress management for college students?"})
print(result2)

# Test Arxiv tool
print("\n3. Testing Arxiv tool (research papers):")
result3 = agent_chain_with_formatting.invoke({"question": "Find recent papers about exercise and mental health in students"})
print(result3)

# Test Medical Research tool
print("\n4. Testing Medical Research tool (PubMed):")
result4 = agent_chain_with_formatting.invoke({"question": "Find medical research about anxiety treatment for college students"})
print(result4)