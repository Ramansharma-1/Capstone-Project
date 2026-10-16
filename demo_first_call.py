"""Week 1 demo — a single LLM call, end to end.

Feature: make one call to the model and read the answer back.
Run alone:  python demo_first_call.py
Run all:    python run.py
"""
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads OPENAI_API_KEY from .env off-Vocareum; harmless no-op on Vocareum
except ImportError:
    pass

client = OpenAI()


def main():
    print("Feature: your first LLM call — one question in, one answer out.\n")
    question = "What is QLORA in one sentence?"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    print(f"Q: {question}")
    print(f"A: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
