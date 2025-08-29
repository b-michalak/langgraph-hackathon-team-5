from typing import List
from typing_extensions import TypedDict


class Address(TypedDict):
    city: str
    zip_code: str
    country: str
    province: str
    address_lines: List[str]


class AddressWithId(Address):
    id: str


class FindInputState(TypedDict):
    address: Address
    description: str


class FindOutputState(TypedDict):
    matchedAddresses: List[AddressWithId]
    newAddress: Address


class WebSearchInputState(TypedDict):
    address: Address


class WebSearchOutputState(TypedDict):
    description: str


class CheckNormalizeInputState(TypedDict):
    address: Address


class NormalizeOutputState(TypedDict):
    normalizedAddress: Address
    description: str
    error: bool


class GraphState(TypedDict):
    address: Address
    normalizedAddress: Address
    matchedAddresses: List[Address]
    description: str
    error: bool
