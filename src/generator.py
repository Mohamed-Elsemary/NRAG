import os
from google import genai
from google.genai import types
from src.indexer import HybridIndexer
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a precise technical assistant for Nokia site engineers.

RULES:
1. Answer strictly using ONLY the provided context blocks.
2. You must append the exact citation to your answer in this format: (Source: Page [Number]). If the answer spans multiple pages, cite all of them.
3. If the provided context does not explicitly contain the answer to the user's question, do not guess. You must reply EXACTLY with: "Not found in the provided document."
"""

class RAGGenerator:
    def __init__(self):
        # Ensure your GEMINI_API_KEY is set in your .env file
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.indexer = HybridIndexer()
        
    def generate_answer(self, query: str) -> str:
        # Fetch Top 4 chunks to ensure we get both slots and RU footprint
        results = self.indexer.search(query, top_k=15)
        
        context_blocks = []
        for res in results:
            chunk = res['chunk']
            p_start = chunk.get('page_start', chunk.get('page', '?'))
            p_end = chunk.get('page_end', p_start)
            page_str = f"{p_start}" if p_start == p_end else f"{p_start}-{p_end}"
            context_blocks.append(f"Page: {page_str}\n{chunk['text']}")
        full_prompt = f"Context:\n{chr(10).join(context_blocks)}\n\nQuestion: {query}"
        
        response = self.client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0
            )
        )
        return response.text.strip()

if __name__ == "__main__":
    rag = RAGGenerator()
    test_q = "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?"
    print(f"\nQuestion: {test_q}")
    print(f"Answer: {rag.generate_answer(test_q)}\n")