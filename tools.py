from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.tools import Tool
from datetime import datetime
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="Useful for when you need to answer questions about current events or general knowledge. Input should be a search query.",
)   

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)
wiki_tool = Tool(
    name="wikipedia_search",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Useful for searching Wikipedia for detailed information about topics, people, places, and historical facts. Input should be a search query.",
)

def save_to_text(data: str, filename: str = "research_output.txt"):
    """
    Save the research data to a text file with a timestamp.
    
    Args:
        data (str): The research data to save.
        filename (str): The name of the file to save the data to.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"---Research Output---\nTimestamp: {timestamp}\n\n{data}\n\n"   
    
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"Data saved to {filename} at {timestamp}."

save_tool = Tool(
    name="save_to_text",
    func=save_to_text,
    description="Saves the research data to a text file with a timestamp. Input should be the research data as a string.",
    )