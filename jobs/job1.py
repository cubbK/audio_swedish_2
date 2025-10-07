import requests
import sys


def test_internet_access():
    try:
        print("Testing internet access...")
        response = requests.get("https://news.ycombinator.com/", timeout=10)

        if response.status_code == 200:
            print(f"✓ Successfully connected to Hacker News!")
            print(f"Status code: {response.status_code}")
            print(f"Content length: {len(response.text)} characters")
            print(f"First 200 characters of response:")
            print("-" * 50)
            print(response.text[:200])
            print("-" * 50)
        else:
            print(f"✗ Connection failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Internet access failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Hello World")
    test_internet_access()
