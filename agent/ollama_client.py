#!/usr/bin/env python3
"""
Ollama client and management utilities
"""

import requests
import subprocess
import time
from typing import Optional


class OllamaManager:
    """Manages Ollama instance and API calls"""

    def __init__(self, model: str = 'phi3', host: str = 'http://localhost:11434'):
        self.model = model
        self.host = host
        self.session = requests.Session()

    def check_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f'{self.host}/', timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list:
        """List available models"""
        try:
            response = self.session.get(f'{self.host}/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []

    def is_model_available(self) -> bool:
        """Check if current model is available"""
        models = self.list_models()
        return any(model.startswith(self.model) for model in models)

    def start_ollama(self) -> bool:
        """Start Ollama server if not running"""
        if self.check_status():
            print('\033[1;32m[Ollama]:\033[0m Already running')
            if not self.is_model_available():
                print(f'\033[1;33m[Warning]:\033[0m Model {self.model} not found')
                print(f'\033[1;33m[Suggestion]:\033[0m Run: ollama pull {self.model}')
                return False
            return True

        print('\033[1;33m[Ollama]:\033[0m Starting server...')
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            time.sleep(3)  # Wait for startup

            if self.check_status():
                print('\033[1;32m[Ollama]:\033[0m Server started successfully')

                # Check if model is available
                if not self.is_model_available():
                    print(f'\033[1;33m[Warning]:\033[0m Model {self.model} not found')
                    print(f'\033[1;33m[Auto-pulling]:\033[0m Attempting to pull {self.model}...')

                    # Try to pull the model
                    try:
                        result = subprocess.run(
                            ["ollama", "pull", self.model],
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minutes timeout
                        )

                        if result.returncode == 0:
                            print(f'\033[1;32m[Success]:\033[0m Model {self.model} pulled successfully')
                            return True
                        else:
                            print(f'\033[1;31m[Error]:\033[0m Failed to pull model: {result.stderr}')
                            return False

                    except subprocess.TimeoutExpired:
                        print(f'\033[1;31m[Error]:\033[0m Model pull timed out')
                        return False
                    except Exception as e:
                        print(f'\033[1;31m[Error]:\033[0m Model pull failed: {str(e)}')
                        return False

                return True
            else:
                print('\033[1;31m[Ollama Error]:\033[0m Failed to start server')
                return False

        except FileNotFoundError:
            print(
                '\033[1;31m[Ollama Error]:\033[0m Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh')
            return False

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response from Ollama"""
        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'system': system_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_ctx': 4096
                }
            }

            response = self.session.post(
                f'{self.host}/api/generate',
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f'\033[1;31m[Ollama Error]: HTTP {response.status_code}\033[0m')
                return ""

        except requests.exceptions.RequestException as e:
            print(f'\033[1;31m[Ollama Error]: {str(e)}\033[0m')
            return ""

    def generate_streaming(self, prompt: str, system_prompt: str = ""):
        """Generate streaming response from Ollama"""
        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'system': system_prompt,
                'stream': True,
                'options': {
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_ctx': 4096
                }
            }

            response = self.session.post(
                f'{self.host}/api/generate',
                json=payload,
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = line.decode('utf-8')
                            import json
                            data = json.loads(chunk)
                            if 'response' in data:
                                yield data['response']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                print(f'\033[1;31m[Ollama Error]: HTTP {response.status_code}\033[0m')

        except requests.exceptions.RequestException as e:
            print(f'\033[1;31m[Ollama Error]: {str(e)}\033[0m')

    def set_model(self, model: str) -> bool:
        """Change the current model"""
        old_model = self.model
        self.model = model

        if self.is_model_available():
            print(f'\033[1;32m[Model Changed]:\033[0m Now using {model}')
            return True
        else:
            print(f'\033[1;33m[Warning]:\033[0m Model {model} not available, reverting to {old_model}')
            self.model = old_model
            return False

    def get_model_info(self) -> dict:
        """Get information about the current model"""
        try:
            response = self.session.post(
                f'{self.host}/api/show',
                json={'name': self.model},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return {}

        except requests.exceptions.RequestException:
            return {}