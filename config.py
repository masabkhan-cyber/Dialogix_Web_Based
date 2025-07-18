import json
import os

CONFIG_PATH = "config.json" # This file can optionally be used for app-wide defaults or admin settings

DEFAULT_CONFIG = {
    "system_prompt": (
        "Your name is Sophia. You are an E-learning Assistant. You communicate only in Urdu, but you must use English text to write (Roman Urdu). Do not use Hindi script or Devanagari. Do not use Urdu script. Only Roman Urdu using English characters is allowed."
    ),
    "whisper_model": "tiny",
    "elevenlabs_api": ""
}

def load_config():
    # This function is now less critical for per-user settings, but can load global defaults if 'config.json' exists.
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return DEFAULT_CONFIG.copy()

def save_config(config):
    # This function is for saving global defaults, if needed.
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)