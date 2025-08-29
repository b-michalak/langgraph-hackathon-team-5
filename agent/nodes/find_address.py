import json
import os
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from models import Address, AddressWithId, FindInputState, FindOutputState

# LLM instance
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

find_address_llm_instructions = """You are provided with a list of addresses and the description about problems with address.
Check if the provided address matches any of the addresses in the list. Apply fuzzy matching and real-world knowledge to find the best match.
Add all matching addresses to the "matchedAddresses" field in the output.
If no addresses match, return an empty list in this field.

Given:
- A target address to find
- A description about problems with address (if none, the field is empty)
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
    {addresses_to_match_against}
</addresses-to-match-against>

If the address to find is incomplete or has errors, use the description to help identify potential matches.
If can not find any matches, return corrected address and empty list of matched addresses.
The matching should be done based on the overall similarity of the address components, not just exact text matches."""


def load_addresses_to_match() -> List[AddressWithId]:
    """Load addresses from the JSON file and add unique IDs."""
    # Get the path to the JSON file relative to this script's parent directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, 'resources', 'addresses-to-match-against.json')

    addresses_with_ids = []

    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate through all countries and their addresses
        for country_code, addresses in data.items():
            for idx, address in enumerate(addresses):
                # Create a unique ID combining country code and index
                address_with_id = AddressWithId(
                    id=f"{country_code}_{idx}",
                    city=address['city'],
                    zip_code=address['zip_code'],
                    province=address['province'],
                    country=address['country'],
                    address_lines=address['address_lines']
                )
                addresses_with_ids.append(address_with_id)

    except FileNotFoundError:
        print(f"Warning: Address file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading addresses: {e}")
        return []

    return addresses_with_ids


def find_address_llm(state: FindInputState):
    """ Find matching addresses from the database """

    city = state['address']['city']
    zip_code = state['address']['zip_code']
    country = state['address']['country']
    province = state['address']['province']
    address_lines = state['address']['address_lines']
    description = state['description']
    addresses_to_match_against = load_addresses_to_match()

    # Enforce structured output
    structured_llm = llm.with_structured_output(FindOutputState)

    # System message
    system_message = find_address_llm_instructions.format(city=city,
                                                          zip_code=zip_code,
                                                          country=country,
                                                          province=province,
                                                          address_lines=", ".join(address_lines),
                                                          description=description,
                                                          addresses_to_match_against=addresses_to_match_against)
    # Generate question
    result = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Find the address based on provided details.")])

    return result
