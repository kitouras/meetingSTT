import os
import json
import requests

project_root_ui_client = os.path.dirname(os.path.abspath(__file__))

class LLMClientWrapper:
    def __init__(self, llm_service_url, api_endpoint, api_key=None, use_auth=False, model_name="gemma-3-4b-it"):
        self.llm_service_url = llm_service_url
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.use_auth = use_auth
        self.model_name = model_name
        self.full_api_url = f"{self.llm_service_url.rstrip('/')}/{self.api_endpoint.lstrip('/')}"

    def summarize(self, text, temperature=0.7, max_tokens=4096):
        if not text:
            print("Error: No text provided for summarization.")
            return None

        template_path = os.path.join(os.path.dirname(project_root_ui_client), "summarize_template.txt")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                user_prompt_template = f.read()
        except FileNotFoundError:
            print(f"Error: summarize_template.txt not found at {template_path}")
            return None
        except Exception as e:
            print(f"Error reading summarize_template.txt: {e}")
            return None

        headers = {"Content-Type": "application/json"}
        if self.use_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.use_auth and not self.api_key:
             print("Warning: LLM API authentication is enabled, but no API key was provided.")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": user_prompt_template.format(text)}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        print(f"Sending text to LLM ({self.model_name}) at {self.full_api_url} for summarization...")
        try:
            response = requests.post(self.full_api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            summary = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            if summary:
                print("Summarization complete.")
                return summary.strip()
            else:
                print("Error: Could not extract summary from LLM response.")
                print("LLM Response:", result)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"LLM Response Status Code: {e.response.status_code}")
                 print(f"LLM Response Text: {e.response.text}")
            return None
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from LLM API.")
            if 'response' in locals() and hasattr(response, 'text'):
                print("Raw Response:", response.text)
            else:
                print("Raw Response: Not available")
            return None
        except Exception as e:
             print(f"An unexpected error occurred during summarization: {e}")
             return None

    def check_health(self):
        health_endpoint = f"{self.llm_service_url.rstrip('/')}/health"
        print(f"Checking LLM service health at {health_endpoint}...")
        try:
            response = requests.get(health_endpoint, timeout=5)
            response.raise_for_status()
            health_data = response.json()
            if health_data.get("status") == "ok":
                print("LLM service is healthy.")
                return {"status": "healthy", "details": health_data}
            else:
                print(f"LLM service health check failed. Status: {health_data.get('status')}, Details: {health_data}")
                return {"status": "unhealthy", "details": health_data}
        except requests.exceptions.RequestException as e:
            print(f"Error checking LLM service health: {e}")
            error_details = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_details += f" | Status Code: {e.response.status_code} | Response: {e.response.text}"
            return {"status": "unreachable", "error": "RequestException", "details": error_details}
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM health check JSON response: {e}")
            raw_response = "Not available"
            if 'response' in locals() and hasattr(response, 'text'):
                raw_response = response.text
            return {"status": "error", "error": "JSONDecodeError", "details": f"Raw response: {raw_response}"}
        except Exception as e:
            print(f"An unexpected error occurred during LLM health check: {e}")
            return {"status": "error", "error": "UnexpectedException", "details": str(e)}