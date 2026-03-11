from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain.tools import tool

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search(query: str) -> str:
    """Search the web for information about the query
    
    Args:
        query (str): The query to search for
    
    Returns:
        str: The search result
    """
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
tools=[search]
agent = create_agent(model=llm,tools=tools)


def main():
    
    print("Test message")
    result = agent.invoke({"messages": [HumanMessage(content="Search 3 job postings for an AI Engineer using langchain worldwide remotely available on Linkedin")]})
    print(result)
    
if __name__ == "__main__":
    main()