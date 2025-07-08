from serpapi import GoogleSearch

params = {
    "q": "Who is the Prime Minister of India?",
    "api_key": "fc7eec8ca8d42ec6bf3044443fb2e8cd1c46c8528f60a131bc16ab29c8fbf7d4"  # Replace with your real key
}

search = GoogleSearch(params)
results = search.get_dict()

# Print the full result to inspect what's inside
import json
print(json.dumps(results, indent=2))
