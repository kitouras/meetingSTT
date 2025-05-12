import requests
import os

class DiarizationServiceClient:
    def __init__(self, service_base_url="http://localhost:5002"):
        """
        Initializes the client for the Diarization and Transcription service.
        Args:
            service_base_url (str): The base URL of the diarization service.
                                     Defaults to http://localhost:5002.
        """
        self.service_base_url = service_base_url
        self.process_audio_url = f"{self.service_base_url}/process_audio"
        self.health_check_url = f"{self.service_base_url}/health"

    def check_health(self):
        """Checks the health of the diarization service."""
        try:
            response = requests.get(self.health_check_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            #print(f"Error checking diarization service health: {e}")
            return {"status": "unreachable", "error": str(e)}

    def process_audio_file(self, audio_file_path):
        """
        Sends an audio file to the diarization service for processing.

        Args:
            audio_file_path (str): The path to the audio file.

        Returns:
            dict: The JSON response from the service (containing transcribed_segments)
                  or None if an error occurred.
        """
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None

        try:
            with open(audio_file_path, 'rb') as f:
                files = {'audio_file': (os.path.basename(audio_file_path), f)}
                print(f"Sending {audio_file_path} to diarization service at {self.process_audio_url}...")
                response = requests.post(self.process_audio_url, files=files)
            
            response.raise_for_status()
            
            result = response.json()
            print("Received response from diarization service.")
            if "transcribed_segments" in result:
                return result
            else:
                print(f"Error: 'transcribed_segments' not in response from diarization service. Response: {result}")
                return result

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred while calling diarization service: {e}")
            if e.response is not None:
                try:
                    print(f"Diarization Service Response (HTTPError): {e.response.json()}")
                    return e.response.json()
                except ValueError:
                    print(f"Diarization Service Response (Raw Text): {e.response.text}")
                    return {"error": "HTTP error from service", "details": e.response.text, "status_code": e.response.status_code}
            return {"error": "HTTP error from service", "details": str(e)}
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: Could not connect to diarization service at {self.process_audio_url}. {e}")
            return {"error": "Connection error to diarization service", "details": str(e)}
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: The request to diarization service timed out. {e}")
            return {"error": "Timeout error with diarization service", "details": str(e)}
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred while calling diarization service: {e}")
            return {"error": "Generic request exception with diarization service", "details": str(e)}
        except Exception as e:
            print(f"An unexpected client-side error occurred: {e}")
            return {"error": "Unexpected client-side error", "details": str(e)}

if __name__ == '__main__':
    client = DiarizationServiceClient()
    
    health = client.check_health()
    print(f"Diarization Service Health: {health}")