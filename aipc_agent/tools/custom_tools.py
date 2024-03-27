from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool

def get_wiki_tool():
    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool(name="Wikipedia",
                      func=wikipedia.run,
	              description="""A useful tool for searching the Internet to find information on world events, 
               						issues, dates, years, etc. Worth using for general topics. Use precise questions.
                     			""")
    return wikipedia_tool

def get_math_tool(problem_chain):
    math_tool = Tool.from_function(name="Calculator",
                func=problem_chain.run,
                 description="""
                 			Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only inputmath expressions.
                    		""")
    return math_tool

def get_word_problem_tool(word_problem_chain):
    word_problem_tool = Tool.from_function(name="Reasoning Tool",
                                       func=word_problem_chain.run,
                                       description="Useful for when you need to answer logic-based/reasoning questions.",
                                    )
    return word_problem_tool