import requests
from requests.auth import HTTPBasicAuth
import json
import os
from dotenv import load_dotenv

load_dotenv()

DOMAIN = os.getenv("JIRA_DOMAIN")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")

def get_transitions(issue_key):
    url = f"{DOMAIN}/rest/api/3/issue/{issue_key}/transitions"
    auth = HTTPBasicAuth(EMAIL, API_TOKEN)
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, auth=auth, headers=headers)
    if response.status_code == 200:
        transitions = response.json().get('transitions', [])
        print(f"Transitions for {issue_key}:")
        for t in transitions:
            print(f"- {t['id']}: {t['name']} (Target: {t['to']['name']})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    get_transitions("FG-10")
