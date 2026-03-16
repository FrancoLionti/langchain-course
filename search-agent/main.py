import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # type: ignore

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


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
agent = create_agent(model=llm,tools=tools, response_format=AgentResponse)


def main():
    
    print("Test message")
    result = agent.invoke({"messages": [HumanMessage(content="Search 3 job postings for an AI Engineer using langchain worldwide remotely available on Linkedin")]})
    print(result)
    
if __name__ == "__main__":
    main()