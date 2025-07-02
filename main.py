from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from tools import search_tool, wiki_tool, save_tool




load_dotenv()

# Define a Pydantic model for the output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]




llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful research assistant that will help generate a research paper.
                  You will use the tools available to you to research the topic and provide a summary, sources, and tools used.
                  
                  IMPORTANT: You must respond with ONLY the JSON format specified below. Do not include any other text, explanations, or commentary.
                  
                  {format_instructions}
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
query = input("Enter your research query: ")
raw_response = agent_executor.invoke({"query": query})
# print(raw_response)

try:
    output = raw_response.get("output")
    
    # Handle if output is a list (new format)
    if isinstance(output, list) and len(output) > 0:
        output_text = output[0].get("text", "")
    else:
        output_text = output
    
    # Try to extract JSON from the output if it contains extra text
    import json
    import re
    
    # Look for JSON pattern in the output
    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        structured_response = parser.parse(json_str)
    else:
        structured_response = parser.parse(output_text)
        
    print("Research Results:")
    print(f"Topic: {structured_response.topic}")
    print(f"Summary: {structured_response.summary}")
    print(f"Sources: {', '.join(structured_response.sources)}")
    print(f"Tools Used: {', '.join(structured_response.tools_used)}")
    
except Exception as e:
    print(f"Error parsing response: {e}")
    print("Raw response:", raw_response)








# def main():
#     print("Hello from ai-agent-tutorial!")


# if __name__ == "__main__":
#     main()
