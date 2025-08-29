from typing import List
import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from models import Address, AddressWithId, CheckNormalizeInputState, NormalizeOutputState


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


def load_new_addresses() -> List[AddressWithId]:
    """Load existing new addresses from the JSON file."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'resources', 'new-addresses.json')

    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing new addresses JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading new addresses: {e}")
        return []


def save_new_address(new_address: Address) -> AddressWithId:
    """Save a new address to the new-addresses.json file and return it with an ID."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'resources', 'new-addresses.json')

    # Load existing new addresses
    existing_addresses = load_new_addresses()

    # Generate a unique ID for the new address
    # Use format: NEW_{country}_{index}
    country_code = new_address['country'].upper()
    existing_count = len([addr for addr in existing_addresses if addr['id'].startswith(f"{country_code}")])
    new_id = f"{country_code}_{existing_count}"

    # Create address with ID
    address_with_id = AddressWithId(
        id=new_id,
        city=new_address['city'],
        zip_code=new_address['zip_code'],
        province=new_address['province'],
        country=new_address['country'],
        address_lines=new_address['address_lines']
    )

    # Add to existing addresses
    existing_addresses.append(address_with_id)

    # Save back to file
    try:
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(existing_addresses, file, indent=2, ensure_ascii=False)
        print(f"Saved new address with ID: {new_id}")
    except Exception as e:
        print(f"Error saving new address: {e}")

    return address_with_id


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

    save_new_address(result['normalizedAddress'])
    return result
