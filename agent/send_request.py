import http.client
import json
import os

def load_addresses(file_path):
    """Load addresses from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def prettify_response(response_text):
    """Parse and prettify JSON response"""
    try:
        # Try to parse as JSON
        response_json = json.loads(response_text)

        # Extract key information for better display
        formatted_response = "📍 Address Processing Result:\n"
        formatted_response += "=" * 70 + "\n"

        # Display original address
        if 'address' in response_json:
            addr = response_json['address']
            formatted_response += "📮 Original Address:\n"
            formatted_response += f"   🏙️  City: {addr.get('city', 'N/A')}\n"
            formatted_response += f"   🗺️  Province: {addr.get('province', 'N/A')}\n"
            formatted_response += f"   🏳️  Country: {addr.get('country', 'N/A')}\n"
            formatted_response += f"   📫 Zip Code: {addr.get('zip_code', 'N/A')}\n"
            formatted_response += f"   📍 Address Lines: {', '.join(addr.get('address_lines', []))}\n"
            formatted_response += "\n"

        # Display normalized address
        if 'normalizedAddress' in response_json:
            norm_addr = response_json['normalizedAddress']
            formatted_response += "✅ Normalized Address:\n"
            formatted_response += f"   🏙️  City: {norm_addr.get('city', 'N/A')}\n"
            formatted_response += f"   🗺️  Province: {norm_addr.get('province', 'N/A')}\n"
            formatted_response += f"   🏳️  Country: {norm_addr.get('country', 'N/A')}\n"
            formatted_response += f"   📫 Zip Code: {norm_addr.get('zip_code', 'N/A')}\n"
            formatted_response += f"   📍 Address Lines: {', '.join(norm_addr.get('address_lines', []))}\n"
            formatted_response += "\n"

        # Display description if available
        if 'description' in response_json:
            formatted_response += f"📝 Description:\n   {response_json['description']}\n\n"

        # Display error status
        if 'error' in response_json:
            status = "❌ Error" if response_json['error'] else "✅ Success"
            formatted_response += f"🔍 Status: {status}\n\n"

        # Display matched addresses if any
        if 'matchedAddresses' in response_json and response_json['matchedAddresses']:
            formatted_response += f"🎯 Matched Addresses: {len(response_json['matchedAddresses'])} found\n\n"

        formatted_response += "-" * 70 + "\n"
        formatted_response += "🔧 Original JSON Response:\n"
        formatted_response += "-" * 70 + "\n"

        # Pretty print the full JSON
        pretty_json = json.dumps(response_json, indent=2, ensure_ascii=False)
        formatted_response += pretty_json
        formatted_response += "\n" + "=" * 70

        return formatted_response
    except json.JSONDecodeError:
        # If it's not JSON, return as is with some formatting
        return f"Raw Response:\n{'-' * 30}\n{response_text}\n{'-' * 30}"

def send_address_request(address_data):
    """Send POST request for a single address"""
    conn = http.client.HTTPConnection("127.0.0.1:2024")

    payload = {
        "assistant_id": "address_data_processing",
        "input": {
            "address": address_data
        },
        "config": {
            "tags": [""],
            "recursion_limit": 10,
            "configurable": {}
        }
    }

    headers = {'content-type': "application/json"}

    try:
        conn.request("POST", "/runs/wait", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        return data.decode("utf-8")
    except Exception as e:
        return f"Error sending request: {e}"
    finally:
        conn.close()

def main():
    # Get the script directory to find addresses.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    addresses_file = os.path.join(script_dir, 'addresses.json')

    # Load addresses from JSON file
    addresses = load_addresses(addresses_file)

    if not addresses:
        print("No addresses found or error loading file")
        return

    print(f"🏠 Found {len(addresses)} addresses to process\n")

    # Process each address
    for i, address in enumerate(addresses, 1):
        print("=" * 70)

        # Send request
        response = send_address_request(address)

        # Prettify and display response
        formatted_response = prettify_response(response)
        print(formatted_response)
        print("\n" + "🟢" * 70 + "\n")

if __name__ == "__main__":
    main()
