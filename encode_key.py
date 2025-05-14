import base64

def encode_api_key(key: str) -> str:
    return base64.b64encode(key.encode()).decode()

if __name__ == "__main__":
    raw_key = input("Enter your OpenAI API key: ").strip()
    encoded = encode_api_key(raw_key)
    print("\ Encoded key (copy this into your .env file):")
    print(f"openai_api_key_encoded={encoded}")
