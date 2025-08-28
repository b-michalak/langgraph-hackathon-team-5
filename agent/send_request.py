import http.client
import json
import os
import sys


def load_addresses(file_path):
    """Load addresses from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


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


def get_user_choice():
    """Get user's choice for which addresses to test"""
    print("🏠 Address Testing Options:")
    print("=" * 50)
    print("1. ✅ Test EXACT MATCHES (should match perfectly)")
    print("2. 🔍 Test FUZZY MATCHES (should match with variations)")
    print("3. ❌ Test NO MATCHES (shouldn't match anything)")
    print("4. ⚠️  Test PROBLEMATIC addresses (missing/invalid data)")
    print("5. 🔄 Test ALL addresses")
    print("6. 📊 Test SPECIFIC address by index")
    print("7. 🚪 Exit")
    print("=" * 50)

    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1-7.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def select_specific_address(addresses_data):
    """Allow user to select a specific address to test"""
    all_addresses = []
    address_labels = []

    # Combine all addresses with labels
    for i, addr in enumerate(addresses_data.get('exact_matches', []), 1):
        all_addresses.append(addr)
        address_labels.append(f"EXACT #{i}: {addr.get('city', 'Unknown')} ({addr.get('country', 'Unknown')})")

    for i, addr in enumerate(addresses_data.get('fuzzy_matches', []), 1):
        all_addresses.append(addr)
        address_labels.append(f"FUZZY #{i}: {addr.get('city', 'Unknown')} ({addr.get('country', 'Unknown')})")

    for i, addr in enumerate(addresses_data.get('no_matches', []), 1):
        all_addresses.append(addr)
        address_labels.append(f"NO MATCH #{i}: {addr.get('city', 'Unknown')} ({addr.get('country', 'Unknown')})")

    for i, addr in enumerate(addresses_data.get('problematic_addresses', []), 1):
        all_addresses.append(addr)
        address_labels.append(f"PROBLEMATIC #{i}: {addr.get('city', 'Unknown')} ({addr.get('country', 'Unknown')})")

    if not all_addresses:
        print("❌ No addresses found!")
        return []

    print(f"\n📋 Available addresses ({len(all_addresses)} total):")
    print("-" * 60)
    for i, label in enumerate(address_labels, 1):
        print(f"{i:2d}. {label}")
    print("-" * 60)

    while True:
        try:
            choice = input(f"\nSelect address (1-{len(all_addresses)}) or 'back' to return: ").strip().lower()
            if choice == 'back':
                return None

            index = int(choice) - 1
            if 0 <= index < len(all_addresses):
                return [all_addresses[index]]
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(all_addresses)} or 'back'.")
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'back'.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def main():
    # Get the script directory to find request-addresses.json in resources folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    addresses_file = os.path.join(script_dir, 'resources', 'request-addresses.json')

    # Load addresses from JSON file
    addresses_data = load_addresses(addresses_file)

    if not addresses_data:
        print("❌ No addresses found or error loading file")
        return

    # Check if the file has the new structure
    if 'exact_matches' not in addresses_data and 'fuzzy_matches' not in addresses_data and 'no_matches' not in addresses_data and 'problematic_addresses' not in addresses_data:
        print("❌ Invalid file format. Expected 'exact_matches', 'fuzzy_matches', 'no_matches', and 'problematic_addresses' sections.")
        return

    exact_count = len(addresses_data.get('exact_matches', []))
    fuzzy_count = len(addresses_data.get('fuzzy_matches', []))
    no_match_count = len(addresses_data.get('no_matches', []))
    problematic_count = len(addresses_data.get('problematic_addresses', []))

    print(f"📊 Loaded addresses: {exact_count} exact matches, {fuzzy_count} fuzzy matches, {no_match_count} no matches, {problematic_count} problematic")

    while True:
        choice = get_user_choice()

        # Initialize variables
        addresses_to_test = []
        test_type = ""

        if choice == '7':  # Exit
            print("👋 Goodbye!")
            break

        elif choice == '6':  # Specific address
            addresses_to_test = select_specific_address(addresses_data)
            if addresses_to_test is None:  # User chose 'back'
                continue
            test_type = "SPECIFIC"

        else:
            # Get addresses based on choice
            if choice == '1':  # Exact matches
                addresses_to_test = addresses_data.get('exact_matches', [])
                test_type = "EXACT MATCHES"
            elif choice == '2':  # Fuzzy matches
                addresses_to_test = addresses_data.get('fuzzy_matches', [])
                test_type = "FUZZY MATCHES"
            elif choice == '3':  # No matches
                addresses_to_test = addresses_data.get('no_matches', [])
                test_type = "NO MATCHES"
            elif choice == '4':  # Problematic addresses
                addresses_to_test = addresses_data.get('problematic_addresses', [])
                test_type = "PROBLEMATIC ADDRESSES"
            elif choice == '5':  # All addresses
                addresses_to_test = addresses_data.get('exact_matches', []) + addresses_data.get('fuzzy_matches', []) + addresses_data.get('no_matches', []) + addresses_data.get('problematic_addresses', [])
                test_type = "ALL"

        if not addresses_to_test:
            print(f"❌ No {test_type.lower()} found!")
            continue

        print(f"\n🚀 Testing {len(addresses_to_test)} {test_type} address(es)...")
        print("=" * 70)

        # Process each address
        for i, address in enumerate(addresses_to_test, 1):
            print(f"\n🏠 Processing address {i}/{len(addresses_to_test)}")
            print("=" * 70)

            # Send request
            response = send_address_request(address)

            # Prettify and display response
            formatted_response = prettify_response(response)
            print(formatted_response)
            print("\n" + "🟢" * 70 + "\n")

        print(f"✅ Completed testing {len(addresses_to_test)} {test_type} address(es)!")

        # Ask if user wants to continue
        while True:
            continue_choice = input("\n🔄 Do you want to test more addresses? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("👋 Goodbye!")
                return
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")


if __name__ == "__main__":
    main()
