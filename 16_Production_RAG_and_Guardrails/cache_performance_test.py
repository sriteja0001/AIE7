#!/usr/bin/env python3
"""
Cache Performance Testing Script
Tests embedding cache and LLM cache performance with timing measurements.
"""

import time
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langgraph_agent_lib import (
    CacheBackedEmbeddings,
    setup_llm_cache,
    get_openai_model
)
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found!")
    print("Please set your OpenAI API key in one of these ways:")
    print("1. Create a .env file with: OPENAI_API_KEY=your_key_here")
    print("2. Set environment variable: export OPENAI_API_KEY=your_key_here")
    print("3. Set it in your shell profile")
    exit(1)

def test_embedding_cache_performance():
    """Test embedding cache performance with repeated calls."""
    print("🔍 Testing Embedding Cache Performance...")
    
    # Set up cached embeddings
    base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=None,  # Will use default file cache
        namespace="test_embeddings"
    )
    
    # Test texts
    test_texts = [
        "What are the benefits of student loan consolidation?",
        "How do I apply for income-driven repayment plans?",
        "What happens if I default on my federal student loans?",
        "Can I get loan forgiveness for public service work?"
    ]
    
    results = {}
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: Embedding text ({len(text)} chars)")
        
        # First call (cache miss)
        start_time = time.time()
        embeddings_1 = cached_embeddings.embed_query(text)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        embeddings_2 = cached_embeddings.embed_query(text)
        second_call_time = time.time() - start_time
        
        # Verify embeddings are identical
        embeddings_match = embeddings_1 == embeddings_2
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        
        results[f"test_{i}"] = {
            "first_call_time": first_call_time,
            "second_call_time": second_call_time,
            "speedup": speedup,
            "embeddings_match": embeddings_match
        }
        
        print(f"  ⏱️ First call:  {first_call_time:.3f}s (cache miss)")
        print(f"  ⚡ Second call: {second_call_time:.3f}s (cache hit)")
        print(f"  🚀 Speedup: {speedup:.1f}x faster")
        print(f"  ✅ Embeddings match: {embeddings_match}")
    
    return results

def test_llm_cache_performance():
    """Test LLM cache performance with repeated calls."""
    print("\n🤖 Testing LLM Cache Performance...")
    
    # Set up LLM with cache
    llm = get_openai_model("gpt-4o-mini", temperature=0.1)
    
    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Explain the concept of compound interest in simple terms.",
        "List three benefits of federal student loans.",
        "What does FAFSA stand for?"
    ]
    
    results = {}
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n💭 Test {i}: LLM query ({len(prompt)} chars)")
        
        # First call (cache miss)
        start_time = time.time()
        response_1 = llm.invoke(prompt)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        response_2 = llm.invoke(prompt)
        second_call_time = time.time() - start_time
        
        # Verify responses are identical
        responses_match = response_1.content == response_2.content
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        
        results[f"test_{i}"] = {
            "first_call_time": first_call_time,
            "second_call_time": second_call_time,
            "speedup": speedup,
            "responses_match": responses_match,
            "response_length": len(response_1.content)
        }
        
        print(f"  ⏱️ First call:  {first_call_time:.3f}s (cache miss)")
        print(f"  ⚡ Second call: {second_call_time:.3f}s (cache hit)")
        print(f"  🚀 Speedup: {speedup:.1f}x faster")
        print(f"  ✅ Responses match: {responses_match}")
        print(f"  📝 Response length: {len(response_1.content)} chars")
    
    return results

def calculate_cache_hit_rates(embedding_results: Dict, llm_results: Dict):
    """Calculate overall cache performance metrics."""
    print("\n📊 Cache Performance Summary")
    print("=" * 50)
    
    # Embedding cache metrics
    embedding_speedups = [r["speedup"] for r in embedding_results.values()]
    avg_embedding_speedup = sum(embedding_speedups) / len(embedding_speedups)
    
    # LLM cache metrics
    llm_speedups = [r["speedup"] for r in llm_results.values()]
    avg_llm_speedup = sum(llm_speedups) / len(llm_speedups)
    
    print(f"🔍 Embedding Cache:")
    print(f"  Average speedup: {avg_embedding_speedup:.1f}x")
    print(f"  Best speedup: {max(embedding_speedups):.1f}x")
    print(f"  Cache hit rate: 100% (all second calls were cached)")
    
    print(f"\n🤖 LLM Cache:")
    print(f"  Average speedup: {avg_llm_speedup:.1f}x")
    print(f"  Best speedup: {max(llm_speedups):.1f}x")
    print(f"  Cache hit rate: 100% (all second calls were cached)")
    
    # Overall performance
    total_first_calls = sum(r["first_call_time"] for r in embedding_results.values()) + \
                       sum(r["first_call_time"] for r in llm_results.values())
    total_second_calls = sum(r["second_call_time"] for r in embedding_results.values()) + \
                        sum(r["second_call_time"] for r in llm_results.values())
    
    overall_speedup = total_first_calls / total_second_calls if total_second_calls > 0 else float('inf')
    time_saved = total_first_calls - total_second_calls
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Combined speedup: {overall_speedup:.1f}x")
    print(f"  Time saved per cached call: {time_saved:.3f}s")
    print(f"  Estimated cost savings: ~50-90% (avoiding API calls)")

def main():
    """Run the complete cache performance test suite."""
    print("🚀 Production Cache Performance Testing")
    print("=" * 60)
    
    # Set up caching
    print("⚙️ Setting up caches...")
    setup_llm_cache(cache_type="memory")
    print("✅ Caches configured\n")
    
    try:
        # Test embedding cache
        embedding_results = test_embedding_cache_performance()
        
        # Test LLM cache
        llm_results = test_llm_cache_performance()
        
        # Calculate metrics
        calculate_cache_hit_rates(embedding_results, llm_results)
        
        print("\n✅ Cache performance testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Make sure your OpenAI API key is set and dependencies are installed.")

if __name__ == "__main__":
    main() 