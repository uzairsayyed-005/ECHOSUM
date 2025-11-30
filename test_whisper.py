#!/usr/bin/env python3

import whisper
import os

def test_whisper():
    print("Testing Whisper with FFmpeg...")
    
    # Check if audiofile.wav exists (created by MoviePy earlier)
    audio_file = "audiofile.wav"
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found. Please run the main script first to create it.")
        return
    
    try:
        # Load whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")  # Using smaller model for testing
        print("Model loaded successfully!")
        
        # Test transcription with a small segment
        print("Testing transcription...")
        result = model.transcribe(audio_file, language="English")
        
        print("SUCCESS! Whisper transcription working properly.")
        print(f"Sample transcription: {result['text'][:100]}...")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_whisper()