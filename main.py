from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main():
    print("Hello from langchain-course!")
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # Invoke the model
    response = llm.invoke("Write a haiku about Python programming.")
    
    print("\nGemini Response:")
    print(response.content)

if __name__ == "__main__":
    main()