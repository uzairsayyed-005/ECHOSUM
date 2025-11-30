#!/usr/bin/env python3
"""
Test script to run the enhanced Meeting Summarizer on a sample file
"""

import os
import sys
from main import transcribe_media, write_transcript, generate_speaker_summary, write_speaker_summary

def test_sample_file():
    """Test the enhanced Meeting Summarizer on a sample file"""
    
    # Test file path
    media_path = "testing/TestVideos/4min_clip_1.mp4"
    num_speakers = 2
    language = "en"
    
    print(f"Testing enhanced Meeting Summarizer on: {media_path}")
    print(f"Expected speakers: {num_speakers}")
    print(f"Language: {language}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(media_path):
        print(f"Error: Test file not found: {media_path}")
        return False
    
    try:
        # Run transcription with enhanced features
        print("Starting transcription with enhanced features...")
        transcription_segments = transcribe_media(media_path, num_speakers, language)
        
        print(f"Transcription completed! Generated {len(transcription_segments)} segments")
        
        # Write transcript
        transcript_path = "test_transcript.txt"
        write_transcript(transcription_segments, transcript_path)
        print(f"Transcript saved to: {transcript_path}")
        
        # Generate speaker summary
        speaker_summary = generate_speaker_summary(transcription_segments)
        summary_path = "test_speaker_summary.txt"
        write_speaker_summary(speaker_summary, summary_path)
        print(f"Speaker summary saved to: {summary_path}")
        
        # Display basic stats
        unique_speakers = list(set(seg.speaker_id for seg in transcription_segments if seg.speaker_id is not None))
        total_duration = max(seg.end for seg in transcription_segments) if transcription_segments else 0
        
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULTS:")
        print("="*50)
        print(f"Total segments: {len(transcription_segments)}")
        print(f"Unique speakers detected: {len(unique_speakers)} (expected: {num_speakers})")
        print(f"Speaker IDs: {unique_speakers}")
        print(f"Total duration: {total_duration:.2f} seconds")
        
        # Show first few segments as preview
        print(f"\nFirst 5 segments preview:")
        for i, segment in enumerate(transcription_segments[:5]):
            speaker_id = segment.speaker_id if segment.speaker_id is not None else "Unknown"
            print(f"  {i+1}. Speaker {speaker_id} ({segment.start:.1f}-{segment.end:.1f}s): {segment.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sample_file()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)