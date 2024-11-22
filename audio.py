import streamlit as st
import pyaudio
import wave
import requests

# Set your OpenAI API Key
OPENAI_API_KEY = "sk-kBI130xMAHzM6cwaJKpY9xORTOnJviq5i_g4IURoVNT3BlbkFJA_E5k7T_o7X1Z2OVwfi3Q_FuT31E8d77rqO9uV"

# Streamlit app UI
#NnYA
st.title("🎤 Real-Time Speech-to-Text with OpenAI Whisper")
st.write("Click the button below to record audio using your microphone, save it locally, and transcribe it to text.")

# Parameters for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 44100  # Hertz
CHUNK = 1024  # Frames per buffer
DURATION = 5  # Duration in seconds
OUTPUT_FILE = "recorded_audio.wav"

# Function to record audio
def record_audio():
    audio = pyaudio.PyAudio()

    st.info("Recording audio...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    st.success("Recording complete!")

    # Save the audio to a file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_FILE

# Button to record audio
if st.button("🎙️ Record Audio"):
    try:
        # Record and save audio
        audio_file_path = record_audio()
        st.audio(audio_file_path, format='audio/wav')
        st.info("Audio saved locally.")

        # Send the WAV file to OpenAI Whisper API
        st.info("Sending to OpenAI Whisper API for transcription...")
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        files = {
            "file": (audio_file_path, open(audio_file_path, "rb")),
            "model": (None, "whisper-1"),
        }

        response = requests.post(url, headers=headers, files=files)

        # Process and display the transcription
        if response.status_code == 200:
            transcription = response.json().get("text")
            st.subheader("Transcription")
            st.write(transcription)
        else:
            st.error("Error during transcription!")
            st.json(response.json())

    except Exception as e:
        st.error(f"An error occurred: {e}")
