from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

### LLM

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


### Schema 

class Result(BaseModel):
    city: str = Field(
        description="City name.",
    )
    zip_code: str = Field(
        description="Zip code.",
    )
    country: str = Field(
        description="Country name.",
    )
    province: str = Field(
        description="Province or state name.",
    )
    address_lines: List[str] = Field(
        description="List of address lines.",
    )
    description: str = Field(
        description="Comment about the address.",
    )

    @property
    def address(self) -> str:
        return f"City: {self.city}\nZip code: {self.zip_code}\nCountry: {self.country}\nProvince: {self.province}\nAddress Lines: {', '.join(self.address_lines)}\nDescription: {self.description}"


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(TypedDict):
    city: str  # City
    zip_code: str  # Zip code
    country: str  # Country
    province: str  # Province
    address_lines: List[str]  # Address lines


### Nodes and edges

find_address_llm_instructions = """You are tasked to find the address. Follow these instructions carefully:

1. First, review the research request details:
{city}, {zip_code}, {country}, {province}, {address_lines}
        
2. Based on the details, identify the most relevant address - review if the provided data are valid.

3. Provide the address as result.

4. In the description field, include any additional context or comments about the address."""


def find_address_llm(state: ResearchGraphState):
    """ Create analysts """

    city = state['city']
    zip_code = state['zip_code']
    country = state['country']
    province = state['province']
    address_lines = state['address_lines']

    # Enforce structured output
    structured_llm = llm.with_structured_output(Result)

    # System message
    system_message = find_address_llm_instructions.format(city=city,
                                                          zip_code=zip_code,
                                                          country=country,
                                                          province=province,
                                                          address_lines=", ".join(address_lines))
    # Generate question 
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Find the address based on provided details.")])

    return {"result": result.address}


normalize_address_instructions = """You are tasked to normalize the address. Follow these instructions carefully:

1. City - Ensure the city name is correctly spelled and capitalized.
2. Zip Code - Verify the zip code format is appropriate for the country.
3. Country - Ensure the country name is correctly spelled and capitalized.
4. Province - Ensure the province or state name is correctly spelled and capitalized.
5. Address Lines - Standardize the address lines by removing any unnecessary punctuation and ensuring proper capitalization
"""


def normalize_address_llm(state: ResearchGraphState):
    """ Create analysts """

    city = state['city']
    zip_code = state['zip_code']
    country = state['country']
    province = state['province']
    address_lines = state['address_lines']

    # Enforce structured output
    structured_llm = llm.with_structured_output(Result)

    # System message
    system_message = normalize_address_instructions
    # Generate question
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(
            content=f"Normalize this address: {city}, {zip_code}, {country}, {province}, {', '.join(address_lines)}")])

    return {"result": result.address}


# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("find_address_llm", find_address_llm)
builder.add_node("normalize_address_llm", normalize_address_llm)

# Logic
builder.add_edge(START, "find_address_llm")
builder.add_edge("find_address_llm", "normalize_address_llm")
builder.add_edge("normalize_address_llm", END)

# Compile
graph = builder.compile()
