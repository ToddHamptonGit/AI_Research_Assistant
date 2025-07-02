from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Define a Pydantic model for the output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the LangChain components
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
    verbose=False  # Set to False for web interface
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Execute the research
        raw_response = agent_executor.invoke({"query": query})
        output = raw_response.get("output")
        
        # Handle if output is a list (new format)
        if isinstance(output, list) and len(output) > 0:
            output_text = output[0].get("text", "")
        else:
            output_text = output
        
        # Try to extract JSON from the output if it contains extra text
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            structured_response = parser.parse(json_str)
        else:
            structured_response = parser.parse(output_text)
        
        # Convert to dictionary for JSON response
        result = {
            'topic': structured_response.topic,
            'summary': structured_response.summary,
            'sources': structured_response.sources,
            'tools_used': structured_response.tools_used
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
