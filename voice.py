import streamlit as st
import torch
from elevenlabs import ElevenLabs, stream as el_stream
from faster_whisper import WhisperModel
import numpy as np
import io
import scipy.io.wavfile as wavfile
import scipy.signal as signal # New import for resampling
import logging
import os
# import librosa # We will try to avoid librosa for now

# Configure logging for better insights
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache the Whisper model to avoid reloading on every rerun
@st.cache_resource
def load_whisper_model(model_name, device, compute_type):
    """Loads the FasterWhisper model and caches it."""
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info(f"Whisper model '{model_name}' loaded successfully on {device}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_name}': {e}")
        st.error(f"Failed to load Whisper model. Please check model name and system resources: {e}")
        return None

def transcribe_audio(audio_bytes, model_name="tiny"):
    """
    Transcribes the audio bytes using faster-whisper.
    This version has VAD and noise reduction explicitly removed/bypassed for debugging,
    and includes explicit resampling to 16kHz using scipy.signal.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8"

    # Load the model using the cached function
    model = load_whisper_model(model_name, device, compute_type)
    if model is None:
        return "" # Model loading failed, cannot proceed

    logger.info(f"🛠️ Using Whisper model: `{model_name}` on `{device}` (compute_type: {compute_type})")
    st.markdown(f"🛠️ Using Whisper model: `{model_name}` on `{device}`") # Keep for Streamlit UI feedback

    if not audio_bytes:
        st.error("Received empty audio data for transcription.")
        logger.error("Received empty audio data for transcription.")
        return ""

    target_sample_rate = 16000 # Whisper models expect 16kHz audio

    try:
        audio_file = io.BytesIO(audio_bytes)
        rate = None
        data = None

        try:
            # Attempt to read the WAV file
            rate, data = wavfile.read(audio_file)
            logger.info(f"Initial WAV read: Sample Rate={rate}Hz, Data Shape={data.shape}, Data Type={data.dtype}")
            if data.size == 0:
                st.error("WAV file read successfully but contains no audio data.")
                logger.error("WAV file read successfully but contains no audio data.")
                return ""

        except ValueError as ve:
            st.error(f"Error reading WAV file (ValueError, often due to corrupted header or non-WAV data): {ve}")
            logger.exception(f"ValueError reading WAV file from bytes: {ve}")
            return ""
        except Exception as e:
            st.error(f"General error reading WAV file from bytes: {e}")
            logger.exception(f"General error reading WAV file from bytes: {e}")
            return ""

        if data.ndim > 1:
            data = data.mean(axis=1) # Convert stereo to mono by averaging channels
            logger.info("Converted multi-channel audio to mono.")

        # --- Data Type Normalization and Conversion to float32 (NumPy 2.0 compatible) ---
        if data.dtype.kind in ('i', 'u'): # Check if data is an integer type (signed or unsigned)
            max_val = np.iinfo(data.dtype).max
            if max_val > 0:
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32) # Fallback for edge cases
        elif data.dtype.kind != 'f': # If not integer and not float already, convert to float32
            data = data.astype(np.float32)
        if data.dtype != np.float32: # Ensure it's explicitly float32, even if it was float64
            data = data.astype(np.float32)

        # --- NOISE REDUCTION BYPASSED FOR DEBUGGING ---
        processed_audio = data # Directly use 'data' without noise reduction
        logger.info("Noise reduction bypassed for debugging.")

        # --- Explicit Resampling to 16kHz using scipy.signal.resample ---
        if rate != target_sample_rate:
            logger.info(f"Resampling audio from {rate}Hz to {target_sample_rate}Hz using scipy.signal.resample...")
            try:
                # num = int(len(processed_audio) * (target_sample_rate / rate))
                # Using up/down factors for more control and sometimes better performance with `resample_poly`
                # Calculate GCD for optimal up/down factors
                gcd_val = np.gcd(target_sample_rate, rate)
                up = target_sample_rate // gcd_val
                down = rate // gcd_val

                processed_audio = signal.resample_poly(processed_audio, up, down)
                rate = target_sample_rate # Update the rate to the new target rate
                logger.info(f"Audio resampled successfully to {target_sample_rate}Hz (scipy.signal). New shape: {processed_audio.shape}")
            except Exception as e:
                st.error(f"Failed to resample audio to {target_sample_rate}Hz using scipy.signal: {e}. Transcription may be affected. Proceeding without explicit resampling.")
                logger.exception(f"Failed to resample audio with scipy.signal: {e}")
                # If resampling fails, proceed with original rate, Faster-Whisper might handle it
                # but results could be suboptimal.

        # Ensure final audio data is float32 and normalized for faster-whisper
        if processed_audio.max() > 1.0 or processed_audio.min() < -1.0:
            max_abs_val = np.max(np.abs(processed_audio))
            if max_abs_val > 0:
                processed_audio = processed_audio / max_abs_val
            logger.info(f"Audio normalized to range [-1.0, 1.0]. Max abs value used: {max_abs_val}")

        processed_audio = processed_audio.astype(np.float32) # Final explicit cast to float32

        # --- Diagnostic Logging Before Transcription ---
        logger.info(f"Pre-transcription audio properties (FINAL): Rate={rate}Hz, Min={np.min(processed_audio):.4f}, Max={np.max(processed_audio):.4f}, Mean={np.mean(processed_audio):.4f}, Std Dev={np.std(processed_audio):.4f}, DType={processed_audio.dtype}, Shape={processed_audio.shape}")

        # --- OPTIONAL: Save the processed audio for manual inspection ---
        output_filename = "processed_audio_for_whisper.wav"
        try:
            wavfile.write(output_filename, rate, processed_audio)
            logger.info(f"Saved processed audio to {output_filename} for debugging (Rate: {rate}Hz).")
            st.info(f"Saved processed audio to `{output_filename}`. Check your app directory.")
        except Exception as e:
            logger.warning(f"Could not save processed audio for debugging: {e}")


        with st.spinner("🎧 Attempting raw transcription (VAD & Noise Reduction disabled, 16kHz resampling)..."):
            segments, info = model.transcribe(
                processed_audio,
                beam_size=5,
                language="en",
                vad_filter=False,
            )

            transcribed_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])

            if not transcribed_text:
                st.warning("Transcription completed, but no discernible text was extracted even without filters and with explicit 16kHz resampling. This strongly points to an issue with: 1) The 'processed_audio_for_whisper.wav' quality (is your voice audible in it?). 2) Microphone input levels/hardware. 3) The 'tiny' Whisper model being too weak. Consider trying a 'base' or 'small' model.", icon="🗣️")
                logger.warning("No text transcribed even with VAD disabled and resampling. This is critical.")
                if not list(segments):
                    logger.warning("Whisper returned no segments at all.")
                else:
                    logger.warning(f"Whisper returned segments, but all were empty after stripping: {[seg.text for seg in segments]}")
                return ""

            logger.info(f"Transcription successful. Text: '{transcribed_text}'")
            return transcribed_text

    except Exception as e:
        st.error(f"🛑 Error during transcription: {e}")
        logger.exception(f"Unhandled error in transcribe_audio: {e}")
        return ""

@st.cache_resource # Changed from @st.cache_data to @st.cache_resource
def get_elevenlabs_client(api_key):
    return ElevenLabs(api_key=api_key)

def speak_text(response_text, api_key):
    """Converts text to speech using ElevenLabs API."""
    if not api_key:
        st.warning("🔇 ElevenLabs API key not set. Please configure it in settings to enable voice responses.")
        logger.warning("ElevenLabs API key not set.")
        return
    if not response_text.strip():
        logger.info("No text to speak from ElevenLabs (empty response).")
        return

    try:
        client = get_elevenlabs_client(api_key=api_key)
        voice_id = "JNaMjd7t4u3EhgkVknn3"
        model_id = "eleven_multilingual_v2"

        logger.info(f"Attempting to speak text with ElevenLabs voice_id={voice_id}, model_id={model_id}")
        audio_stream = client.text_to_speech.stream(
            text=response_text,
            voice_id=voice_id,
            model_id=model_id
        )
        el_stream(audio_stream)
        logger.info("ElevenLabs audio stream played successfully.")

    except Exception as e:
        st.error(f"🛑 TTS Error: {e}. Check your ElevenLabs API key and network connection. Also verify voice_id/model_id.")
        logger.exception(f"Error in speak_text (ElevenLabs): {e}")