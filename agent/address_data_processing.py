from typing import List

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict, Literal

### LLM

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class Address(TypedDict):
    city: str  # City
    zip_code: str  # Zip code
    country: str  # Country
    province: str  # Province
    address_lines: List[str]


class AddressWithId(Address):
    id: str  # Unique identifier from the database


class GraphState(TypedDict):
    address: Address
    normalizedAddress: Address
    matchedAddresses: List[Address]
    description: str
    error: bool


class WebSearchInputState(TypedDict):
    address: Address


class WebSearchOutputState(TypedDict):
    description: str


class FindInputState(TypedDict):
    address: Address
    description: str


class FindOutputState(TypedDict):
    matchedAddresses: List[AddressWithId]


class CheckNormalizeInputState(TypedDict):
    address: Address


class NormalizeOutputState(TypedDict):
    normalizedAddress: Address
    description: str
    error: bool


### Nodes and edges

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


find_address_llm_instructions = """You are provided with a list of addresses and the description about it from the web search.
Check if the provided address matches any of the addresses in the list. Apply fuzzy matching and real-world knowledge to find the best match.
Add all matching addresses to the "matchedAddresses" field in the output.
If no addresses match, return an empty list in this field.

Given:
- A target address to find
- A description about the address to find from web search
- A list of candidate addresses to match against

<address-to-find>
    <city>{city}</city>
    <zip_code>{zip_code}</zip_code>
    <country>{country}</country>
    <province>{province}</province>
    <address_lines>{address_lines}</address_lines>
</address-to-find>

<description>
    {description}
<description/>

<addresses-to-match-against>
    <address>
        <id>1</id>
        <city>Warszawa</city>
        <zip_code>01-234</zip_code>
        <country>PL</country>
        <province>mazowieckie</province>
        <address_lines>Pl. Konstytucji 12/3</address_lines>
    </address>
    <address>
        <id>2</id>
        <city>Kraków</city>
        <zip_code>31-001</zip_code>
        <country>PL</country>
        <province>małopolskie</province>
        <address_lines>Wawel 1</address_lines>
    </address>
    <address>
        <id>3</id>
        <city>Lublin</city>
        <zip_code>20-810</zip_code>
        <country>PL</country>
        <province>lubelskie</province>
        <address_lines>Ul. Spokojna 123</address_lines>
    </address>
</addresses-to-match-against>
"""


def find_address_llm(state: FindInputState):
    """ Create analysts """

    city = state['address']['city']
    zip_code = state['address']['zip_code']
    country = state['address']['country']
    province = state['address']['province']
    address_lines = state['address']['address_lines']
    description = state['description']

    # Enforce structured output
    structured_llm = llm.with_structured_output(FindOutputState)

    # System message
    system_message = find_address_llm_instructions.format(city=city,
                                                          zip_code=zip_code,
                                                          country=country,
                                                          province=province,
                                                          address_lines=", ".join(address_lines),
                                                          description=description)
    # Generate question
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Find the address based on provided details.")])

    return result


def was_address_found(state: GraphState) -> Literal[END, "check_normalize_address_llm"]:
    """Check if any address was found"""
    if len(state['matchedAddresses']) > 0:
        return END
    else:
        return "check_normalize_address_llm"


check_normalize_address_llm_instructions = """Your task is checking and normalizing the address provided below.
The input address is:
<address>
    <city>{city}</city>
    <zip_code>{zip_code}</zip_code>
    <country>{country}</country>
    <province>{province}</province>
    <address_lines>{address_lines}</address_lines>
</address>

Follow these instructions carefully:

- Put the normalized address in the "normalizedAddress" field in the output.
- Check if country is a valid country ISO 2-letter code. If not, set the "error" field to "true".
- Check if the city name is correctly spelled and capitalized. Check that the city exists in the specified country or if you are not aware of a city of such name check if there is a well-known city with a similar name and correct it.
- If the city name is a translated version, use the name that is commonly used in the specified country. For example, for country "PL", replace "Warsaw" with "Warszawa".
- If the city name does not resemble any existing city in the specified country, leave it as is.
- Verify the zip code format is appropriate for the country. Normalize it if it differs from the desired format only due to missing hyphens or similar formatting issues but the type and number of characters is otherwise correct.
- Ensure the province or state name is correctly spelled and capitalized. Check that the province exists in the specified country.
- Standardize the address lines by removing any unnecessary punctuation and ensuring proper capitalization.
- For Polish addresses, normalize the street address to contain the abbreviated street type, for example normalize "Cicha 27" to "Ul. Cicha 27" and "Plac Litewski 2/3" to "Pl. Litewski 2/3".
- In the description field, include any additional context or comments about the address and its normalization.
- If the address does not match the pattern of addresses found in the specified country, set the "error" field to "true".
"""


def check_normalize_address_llm(state: CheckNormalizeInputState):
    """Check the address for correctness and normalize it if possible"""

    city = state['address']['city']
    zip_code = state['address']['zip_code']
    country = state['address']['country']
    province = state['address']['province']
    address_lines = state['address']['address_lines']

    # Enforce structured output
    structured_llm = llm.with_structured_output(NormalizeOutputState)

    # System message
    system_message = check_normalize_address_llm_instructions.format(city=city,
                                                                     zip_code=zip_code,
                                                                     country=country,
                                                                     province=province,
                                                                     address_lines=", ".join(address_lines))
    # Generate question
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [
            HumanMessage(content="Validate and normalize the address based on provided details.")])

    return result


# Add nodes and edges
builder = StateGraph(GraphState)
builder.add_node("web_search_address", web_search_address)
builder.add_node("find_address_llm", find_address_llm)
builder.add_node("check_normalize_address_llm", check_normalize_address_llm)

# Logic
builder.add_edge(START, "web_search_address")
builder.add_edge("web_search_address", "find_address_llm")
builder.add_conditional_edges("find_address_llm", was_address_found)
builder.add_edge("check_normalize_address_llm", END)

# Compile
graph = builder.compile()
