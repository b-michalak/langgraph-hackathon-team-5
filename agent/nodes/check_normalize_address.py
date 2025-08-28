from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict

# Type definitions
class Address(TypedDict):
    city: str
    zip_code: str
    country: str
    province: str
    address_lines: List[str]

class CheckNormalizeInputState(TypedDict):
    address: Address

class NormalizeOutputState(TypedDict):
    normalizedAddress: Address
    description: str
    error: bool

# LLM instance
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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
