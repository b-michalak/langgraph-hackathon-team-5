from typing import List
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict

# Type definitions
class Address(TypedDict):
    city: str
    zip_code: str
    country: str
    province: str
    address_lines: List[str]

class WebSearchInputState(TypedDict):
    address: Address

class WebSearchOutputState(TypedDict):
    description: str

# LLM instance
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

web_search_address_instructions = """You are an AI assistant specializing in finding information about the address data.

Your task is to take a information from the web search about the address and return the description about it.

The input is a list of web search results:
<formatted_web_search_info>
    {formatted_web_search_info}
<formatted_web_search_info/>

The address to validate is:
<address>
    <city>{city}</city>
    <zip_code>{zip_code}</zip_code>
    <country>{country}</country>
    <province>{province}</province>
    <address_lines>{address_lines}</address_lines>
</address>

As output provide a description field with any additional context or comments about the address and its validation."""


def build_combined_address(address_lines: List[str], city: str, province: str, zip_code: str, country: str) -> str:
    """Build a combined address string from components."""
    components = [", ".join(address_lines), city]

    if province:
        components.append(province)

    components.extend([zip_code, country])

    return ", ".join(components)


def web_search_address(state: WebSearchInputState):
    """ Retrieve information from web search """
    city = state['address']['city']
    zip_code = state['address']['zip_code']
    country = state['address']['country']
    province = state['address']['province']
    address_lines = state['address']['address_lines']
    combined_address = build_combined_address(address_lines, city, province, zip_code, country)

    tavily_search = TavilySearchResults(max_results=3)
    web_search_info = tavily_search.invoke(combined_address)

    # If web_search_info is not a string or proper list, default to empty string
    if not isinstance(web_search_info, (str, list)):
        web_search_info = ""

    if isinstance(web_search_info, str):
        formatted_web_search_info = web_search_info
    else:
        # web_search_info is a list of dictionaries
        formatted_web_search_info = "\n\n---\n\n".join(
            [
                f'<document href="{doc["url"]}"/>\n{doc["content"]}\n</document>'
                for doc in web_search_info
            ]
        ) if web_search_info else ""

    system_message = web_search_address_instructions.format(formatted_web_search_info=formatted_web_search_info,
                                                            city=city,
                                                            zip_code=zip_code,
                                                            country=country,
                                                            province=province,
                                                            address_lines=", ".join(address_lines))

    structured_llm = llm.with_structured_output(WebSearchOutputState)
    result = structured_llm.invoke([SystemMessage(content=system_message)])

    return result
