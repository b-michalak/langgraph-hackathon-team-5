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
    error: bool = Field(
        description="Indicates if there was an error in processing the address.",
    )

    @property
    def address(self) -> str:
        return f"error: {self.error}\nCity: {self.city}\nZip code: {self.zip_code}\nCountry: {self.country}\nProvince: {self.province}\nAddress Lines: {', '.join(self.address_lines)}\nDescription: {self.description}"


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class Address(TypedDict):
    city: str  # City
    zip_code: str  # Zip code
    country: str  # Country
    province: str  # Province
    address_lines: List[str]

class GraphState(TypedDict):
    address: Address
    matchedAddresses: List[Address]
    result: Result

class InputState(TypedDict):
    address: Address

class FindOutputState(TypedDict):
    matchedAddresses: List[Address]

class CheckNormalizeInputState(TypedDict):
    address: Address

class OutputState(TypedDict):
    normalizedAddress: Address
    description: str
    error: bool


### Nodes and edges

find_address_llm_instructions = """You are tasked to find the address. Follow these instructions carefully:

1. First, review the research request details:
{city}, {zip_code}, {country}, {province}, {address_lines}
        
2. Based on the details, identify the most relevant address - review if the provided data are valid.

3. Provide the address as result.

4. In the description field, include any additional context or comments about the address."""


def find_address_llm(state: InputState):
    """ Create analysts """

    city = state['address']['city']
    zip_code = state['address']['zip_code']
    country = state['address']['country']
    province = state['address']['province']
    address_lines = state['address']['address_lines']

    # Enforce structured output
    structured_llm = llm.with_structured_output(FindOutputState)

    # System message
    system_message = find_address_llm_instructions.format(city=city,
                                                          zip_code=zip_code,
                                                          country=country,
                                                          province=province,
                                                          address_lines=", ".join(address_lines))
    # Generate question
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Find the address based on provided details.")])

    return result



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

- Check if country is a valid country ISO 2-letter code. If not, set the "error" field to "true".
- Check if the city name is correctly spelled and capitalized. Check that the city exists in the specified country or if you are not aware of a city of such name check if there is a well-known city with a similar name and correct it.
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
    structured_llm = llm.with_structured_output(OutputState)

    # System message
    system_message = check_normalize_address_llm_instructions.format(city=city,
                                                                     zip_code=zip_code,
                                                                     country=country,
                                                                     province=province,
                                                                     address_lines=", ".join(address_lines))
    # Generate question 
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Find the address based on provided details.")])

    return result

# Add nodes and edges
builder = StateGraph(GraphState)
builder.add_node("find_address_llm", find_address_llm)
builder.add_node("check_normalize_address_llm", check_normalize_address_llm)

# Logic
builder.add_edge(START, "find_address_llm")
builder.add_edge("find_address_llm", "check_normalize_address_llm")
builder.add_edge("check_normalize_address_llm", END)

# Compile
graph = builder.compile()
