"""Main script to run the application services.

This script manages the lifecycle of the application, including:
- Starting and stopping Docker Compose services for the backend.
- Automatically rebuilding the Docker image if source files change.
- Starting and stopping the Python-based UI client.
- Handling graceful shutdown on signals like Ctrl+C.
"""
import subprocess
import signal
import sys
import os
import time
import atexit
import json
import hashlib
from types import FrameType
from typing import Optional, Dict, Any

from python_on_whales import DockerClient, DockerException


BUILD_STATE_FILE = ".diarization_build_state.json"


def get_file_hash(filepath: str) -> Optional[str]:
    """Computes the SHA256 hash of a file.

    Args:
        filepath: The path to the file.

    Returns:
        The hex digest of the file's hash, or None if the file doesn't exist.
    """
    h = hashlib.sha256()
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_dir_hash(dirpath: str) -> Optional[str]:
    """Computes a representative SHA256 hash for a directory's contents.

    Args:
        dirpath: The path to the directory.

    Returns:
        A single hash representing the contents of the directory, or None if
        the directory doesn't exist.
    """
    if not os.path.isdir(dirpath):
        return None
    hashes = []
    for root, _, files in os.walk(dirpath):
        for name in sorted(files):
            filepath = os.path.join(root, name)
            file_hash = get_file_hash(filepath)
            if file_hash:
                hashes.append(file_hash)

    dir_hasher = hashlib.sha256()
    for h_val in sorted(hashes):
        dir_hasher.update(h_val.encode('utf-8'))
    return dir_hasher.hexdigest()


def get_current_source_state() -> Dict[str, Optional[str]]:
    """Gets the current hashes of all watched source files and directories.

    Returns:
        A dictionary mapping file/dir paths to their content hashes.
    """
    state: Dict[str, Optional[str]] = {}
    source_files_to_hash = ["Dockerfile"]
    source_dirs_to_hash = ["diarization_service"]

    for f_path in source_files_to_hash:
        state[f_path] = get_file_hash(f_path)

    for d_path in source_dirs_to_hash:
        state[d_path] = get_dir_hash(d_path)

    return state


def load_previous_source_state() -> Dict[str, Any]:
    """Loads the last saved source state from the state file.

    Returns:
        A dictionary with the previously saved source state, or an empty dict.
    """
    if os.path.exists(BUILD_STATE_FILE):
        try:
            with open(BUILD_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {BUILD_STATE_FILE}. Assuming no previous state.")
            return {}
    return {}


def save_source_state(state: Dict[str, Optional[str]]) -> None:
    """Saves the current source state to the state file.

    Args:
        state: The current source state dictionary to save.
    """
    try:
        with open(BUILD_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        print(f"Build state saved to {BUILD_STATE_FILE}")
    except IOError as e:
        print(f"Error saving build state to {BUILD_STATE_FILE}: {e}")


ui_process: Optional[subprocess.Popen] = None
docker: Optional[DockerClient] = None


def get_project_name() -> str:
    """Gets the project name from the directory name or uses a default.

    Returns:
        A sanitized string to be used as the Docker Compose project name.
    """
    return os.path.basename(os.getcwd()).lower().replace("_", "").replace("-", "") or "meetingstt"


def start_docker_services() -> None:
    """Starts the Docker Compose services.

    Checks if a rebuild is needed based on source file changes before starting.
    """
    global docker
    print("Initializing Docker client...")
    try:
        compose_file_path = os.path.join(os.getcwd(), "docker-compose.yml")
        if not os.path.exists(compose_file_path):
            print(f"Error: docker-compose.yml not found at {compose_file_path}")
            sys.exit(1)

        project_name = get_project_name()
        print(f"Using project name for Docker Compose: {project_name}")
        docker = DockerClient(compose_project_name=project_name, compose_files=[compose_file_path])

        project_name = get_project_name()
        diarization_image_name_base = f"{project_name}-diarization_service"
        diarization_image_to_check = f"{diarization_image_name_base}:latest"

        perform_build = False
        try:
            docker.image.inspect(diarization_image_to_check)
            print(f"Image {diarization_image_to_check} found.")
            current_state = get_current_source_state()
            previous_state = load_previous_source_state()

            if current_state != previous_state:
                print("Source files for diarization_service have changed. Rebuild is required.")
                perform_build = True
            else:
                print("Source files for diarization_service unchanged. No rebuild needed based on source changes.")
        except DockerException as e:
            if "No such image" in str(e) or (hasattr(e, 'reason') and "No such image" in e.reason):
                print(f"Image {diarization_image_to_check} not found. Build is required.")
                perform_build = True
            else:
                print(f"Could not inspect image {diarization_image_to_check} (Reason: {e}). Assuming build is required.")
                perform_build = True
        except Exception as e:
            print(f"An unexpected error occurred while checking image/sources: {e}. Assuming build is required.")
            perform_build = True


        if perform_build:
            print("Starting Docker Compose services with build enabled for diarization_service (docker-compose up --build -d)...")
        else:
            print("Starting Docker Compose services with build disabled for diarization_service (docker-compose up --no-build -d)...")
        
        docker.compose.up(detach=True, build=perform_build, remove_orphans=True)
        print("Docker Compose services command executed.")

        if perform_build:
            try:
                docker.image.inspect(diarization_image_to_check)
                current_state_after_build = get_current_source_state()
                save_source_state(current_state_after_build)
            except DockerException:
                print(f"Warning: diarization_service image {diarization_image_to_check} still not found after build attempt. State not saved.")
            except Exception as e:
                print(f"Warning: Could not save build state after build: {e}")

        print("Waiting for services to become healthy (checking diarization_service)...")
        time.sleep(10)
        
        max_retries = 12
        retries = 0
        diarization_service_name = f"{project_name}-diarization_service-1"

        try:
            running_containers = docker.compose.ps()
            found_service = None
            for container in running_containers:
                if "diarization_service" in container.name.lower():
                    found_service = container
                    diarization_service_name = container.name
                    break
            if not found_service:
                 print(f"Warning: Could not dynamically find diarization_service container. Using default: {diarization_service_name}")

        except Exception as e:
            print(f"Warning: Could not list compose services to find diarization_service: {e}. Using default: {diarization_service_name}")


        while retries < max_retries:
            try:
                container = docker.container.inspect(diarization_service_name)
                if container.state.running and container.state.health and container.state.health.status == "healthy":
                    print(f"Service {diarization_service_name} is healthy.")
                    break
                elif container.state.running:
                    print(f"Service {diarization_service_name} is running, waiting for health check... (Status: {container.state.status}, Health: {container.state.health.status if container.state.health else 'N/A'})")
                else:
                    print(f"Service {diarization_service_name} is not running (State: {container.state.status}).")
                    try:
                        logs = docker.container.logs(diarization_service_name, tail=20)
                        print(f"Last 20 lines of logs for {diarization_service_name}:\n{logs}")
                    except Exception as log_e:
                        print(f"Could not fetch logs for {diarization_service_name}: {log_e}")
            except DockerException as e:
                print(f"Error inspecting service {diarization_service_name}: {e}. Retrying...")
            except Exception as e:
                print(f"Unexpected error checking service {diarization_service_name} health: {e}. Retrying...")
            
            retries += 1
            time.sleep(10)
        
        if retries == max_retries:
            print(f"Warning: Service {diarization_service_name} did not become healthy after {max_retries * 10} seconds.")
            try:
                logs = docker.container.logs(diarization_service_name, tail=50)
                print(f"Last 50 lines of logs for {diarization_service_name} on timeout:\n{logs}")
            except Exception:
                pass

    except DockerException as e:
        print(f"Docker Compose operation failed: {e}")
        try:
            if docker:
                services = docker.compose.ps()
                for service in services:
                    print(f"\nLogs for {service.name} (last 20 lines):")
                    print(docker.container.logs(service.name, tail=20))
        except Exception as log_e:
            print(f"Could not retrieve service logs: {log_e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during Docker service startup: {e}")
        sys.exit(1)

def stop_docker_services() -> None:
    """Stops the Docker Compose services."""
    global docker
    if docker:
        print("Stopping Docker Compose services (docker-compose down)...")
        try:
            docker.compose.down(remove_orphans=True, quiet=False)
            print("Docker Compose services stopped.")
        except DockerException as e:
            print(f"Error stopping Docker Compose services: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during Docker service shutdown: {e}")
    else:
        print("Docker client not initialized, skipping Docker stop.")


def start_ui_client() -> None:
    """Starts the UI client application as a separate process."""
    global ui_process
    print("Starting UI client (python -m ui_client.api)...")
    try:
        env = os.environ.copy()
        ui_process = subprocess.Popen(
            [sys.executable, "-m", "ui_client.api"],
            env=env
        )
        print(f"UI client process initiated with PID: {ui_process.pid}. Checking for immediate exit...")
        time.sleep(3)
        if ui_process.poll() is not None:
            print(f"Error: UI client process (PID: {ui_process.pid}) exited prematurely with code {ui_process.poll()}.")
            print("Check the console output from the UI client for more details.")
            handle_exit(None, None)
        else:
            print(f"UI client (PID: {ui_process.pid}) appears to be running.")
    except FileNotFoundError:
        print("Error: Python executable not found or ui_client.api module not found.")
        print("Make sure you are in the project root directory and ui_client is a runnable module.")
        handle_exit(None, None)
    except Exception as e:
        print(f"An unexpected error occurred while trying to start the UI client: {e}")
        handle_exit(None, None)


def handle_exit(sig: int, frame: Optional[FrameType]) -> None:
    """Handles script termination signals for graceful shutdown.

    Args:
        sig: The signal number received.
        frame: The current stack frame.
    """
    print(f"\nSignal {sig} received. Shutting down...")

    global ui_process
    if ui_process and ui_process.poll() is None:
        print("Terminating UI client process...")
        ui_process.terminate()
        try:
            ui_process.wait(timeout=10)
            print("UI client process terminated.")
        except subprocess.TimeoutExpired:
            print("UI client process did not terminate gracefully, killing...")
            ui_process.kill()
            print("UI client process killed.")
    
    if docker:
        stop_docker_services()
    
    print("Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    atexit.register(stop_docker_services)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGBREAK, handle_exit)

    print("--- Starting Application ---")
    start_ui_client()

    if ui_process is None or ui_process.poll() is not None:
        print("UI client failed to start or exited prematurely. run.py will now exit.")
        sys.exit(1)

    print("UI client process appears to be running. Proceeding to start Docker services...")
    start_docker_services()

    print(f"Now waiting for UI client process (PID: {ui_process.pid}) to complete...")
    try:
        ui_process.wait()
        print(f"UI client process (PID: {ui_process.pid}) has exited with code: {ui_process.returncode}.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received while waiting for UI process. Signal handler will manage shutdown.")
    except Exception as e:
        print(f"An unexpected error occurred while waiting for the UI process: {e}")
    finally:
        print("--- Application Shutdown Sequence Initiated (or UI exited) ---")

    print("run.py script finished.")