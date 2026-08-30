import time
from src.generator import RAGGenerator
QUESTIONS = [
    "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?",
    "What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?",
    "What are the two software load-lines supported by the 1830 PSS system?",
    "Which fan units are supported on the 1830 PSS-32 shelf?",
    "Which fan unit(s) are used on the 1830 PSS-16II shelf?",
    "Name the power filter cards supported on the 1830 PSS-8 shelf.",
    "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported?",
    "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?"
]

def run_evaluation():
    rag = RAGGenerator()
    print(f"{'Q#':<3} | {'Answer'}")
    print("-" * 120)
    
    for i, q in enumerate(QUESTIONS, 1):
        try:
            ans = rag.generate_answer(q).replace('\n', ' ')
            print(f"{i:<3} | {ans}")
        except Exception as e:
            print(f"{i:<3} | ERROR: {str(e)}")
            
        if i < len(QUESTIONS):
            time.sleep(15)

if __name__ == "__main__":
    run_evaluation()