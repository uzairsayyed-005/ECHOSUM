import os
import numpy as np
from dataclasses import dataclass
import whisper
import torch
import re

AS_SEGMENT_DURATION = 0.1  # Duration of each segment in the speaker diarization, in seconds
TRANSCRIPT_PATH = 'transcript.txt'  # Path to the (final) transcript file (after speaker naming)
SPEAKER_SUMMARY_PATH = 'speaker_summary.txt'  # Path to the speaker-wise summary file

# Global model variable to cache the Whisper model
_whisper_model = None

def standardize_transcript_format(transcript_text):
    """Standardize transcript format for better evaluation accuracy"""
    lines = []
    for line in transcript_text.split('\n'):
        line = line.strip()
        if line:
            # Standardize timestamp format: [MM:SS-MM:SS]
            line = re.sub(r'\((\d+\.\d+)-(\d+\.\d+)\)', r'[\1-\2]', line)
            # Standardize speaker format: Speaker_X:
            line = re.sub(r'Speaker (\d+):', r'Speaker_\1:', line)
            line = re.sub(r'Speaker (\w+):', r'Speaker_\1:', line)
            # Remove double spaces
            line = re.sub(r'\s+', ' ', line)
            lines.append(line)
    return '\n'.join(lines)

def normalize_text_for_evaluation(text):
    """Normalize text for better character-level comparison"""
    # Convert to lowercase for comparison
    text = text.lower()
    # Remove punctuation inconsistencies
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize common word variations
    replacements = {
        "gonna": "going to",
        "wanna": "want to", 
        "yeah": "yes",
        "um": "",
        "uh": "",
        "like": "",  # Remove filler words
        "you know": "",
        "i mean": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_whisper_model():
    """Get or load the Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {device}')
        print('Loading Whisper model (this happens only once)...')
        _whisper_model = whisper.load_model('base', device=device)
        print('Whisper model loaded and cached')
    return _whisper_model


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: int | None  # -1 for silence


def transcribe_media(media_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the media file (audio or video) to text. The function extracts audio from video files or processes audio files directly, performs speaker diarization and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment.
    :param media_path: Path to the audio or video file
    :param num_speakers: Number of speakers in the audio/video file
    :param language: Language of the audio/video file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    import subprocess
    import json
    
    tmp_audio_path = 'audiofile.wav'  # Path to the temporary audio file
    
    # Check if file is audio or video using ffprobe
    def is_video_file(file_path):
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                return any(stream.get('codec_type') == 'video' for stream in streams)
        except:
            pass
        return False
    
    # Determine if this is a video or audio file
    is_video = is_video_file(media_path)
    
    if is_video:
        # Handle video files using moviepy
        from moviepy.editor import VideoFileClip
        print(f'Converting video {media_path} to {tmp_audio_path}')
        try:
            video = VideoFileClip(media_path)
            print(f'Loaded video {media_path}')
            if video.audio is None:
                raise ValueError('Video has no audio')
            # Optimized audio extraction: lower sample rate for faster processing
            video.audio.write_audiofile(tmp_audio_path, codec='pcm_s16le', fps=16000, nbytes=2, verbose=False, logger=None)
            video.close()
            print('Converted video to audio')
        except Exception as e:
            print(f'Error processing video file: {e}')
            print('Falling back to ffmpeg for audio extraction...')
            # Fallback to ffmpeg for problematic video files
            cmd = ['ffmpeg', '-y', '-i', media_path, '-ar', '16000', '-ac', '2', '-c:a', 'pcm_s16le', tmp_audio_path]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise ValueError(f'Failed to extract audio from video: {media_path}')
            print('Extracted audio using ffmpeg')
    else:
        # Handle audio files directly using ffmpeg
        print(f'Converting audio {media_path} to {tmp_audio_path}')
        cmd = ['ffmpeg', '-y', '-i', media_path, '-ar', '16000', '-ac', '2', '-c:a', 'pcm_s16le', tmp_audio_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise ValueError(f'Failed to process audio file: {media_path}')
        print('Converted audio to required format')

    segments = transcribe_audio(tmp_audio_path, num_speakers, language)

    os.remove(tmp_audio_path)

    return segments


def transcribe_audio(audio_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the audio file to text. The function performs speaker diarization on the audio file and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment. These segments should represent the dialogue as accurately as possible, with each segment containing the text spoken by a single speaker and the next segment containing the text spoken by another speaker. The speaker ID should be unique for each speaker and should be consistent throughout the transcription.
    :param audio_path: Path to the audio file
    :param num_speakers: Number of speakers in the audio file
    :param language: Language of the audio file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    # Use the cached Whisper model
    model = get_whisper_model()
    print('Loaded model')

    # Transcribe the audio
    print('Transcribing audio')
    result = model.transcribe(audio_path, language=language, verbose=True)
    print('Transcribed audio')

    segments = []
    for segment in result['segments']:
        segments.append(Segment(segment['start'], segment['end'], segment['text'].strip(), None))  # type: ignore

    print(segments)

    flags = __diarize_audio(audio_path, num_speakers)

    print('Flags')
    print(flags)

    # Map diarization speaker flags to Whisper transcription segments
    transcription_segments = __map_sentences_to_speakers(segments, flags)

    print(transcription_segments)

    # Organize the transcription by speaker
    organized_segments = __organize_by_speaker(transcription_segments)
    print(organized_segments)

    return organized_segments


def write_transcript(segments: list[Segment], output_path: str) -> None:
    """
    Write the transcription segments to the console and a text file. Each segment is printed with the speaker ID and text. The output file will contain the same information. The output file is opened in the default text editor after writing.
    """

    for segment in segments:
        print(f'Speaker {segment.speaker_id} ({segment.start:.2f}-{segment.end:.2f}): {segment.text}')

    unique_speakers = len(set([segment.speaker_id for segment in segments]))
    print(f'In total we have {len(segments)} segments and {unique_speakers} unique speakers')

    # Generate transcript content
    transcript_lines = []
    for segment in segments:
        transcript_lines.append(f'Speaker {segment.speaker_id} ({segment.start:.2f}-{segment.end:.2f}): {segment.text}')
    
    transcript_content = '\n'.join(transcript_lines)
    
    # Apply standardization for better evaluation
    standardized_content = standardize_transcript_format(transcript_content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(standardized_content)

    os.startfile(output_path)  # open the file in the default file editor


def generate_speaker_summary(segments: list[Segment]) -> dict:
    """Aggregate speaker-wise statistics and return a summary structure.

    Returns a dictionary keyed by speaker label (name or ID) with:
      total_time: float (seconds spoken)
      segment_count: int
      word_count: int
      avg_words_per_segment: float
      speaking_share: float (percentage of total speaking time)
      first_ts: float (first appearance timestamp)
      last_ts: float (last appearance timestamp)
      top_keywords: list[str]
      sample_excerpt: str
    """
    from collections import defaultdict, Counter
    import re

    # Collect per-speaker text and timing
    speaker_data: dict = {}
    total_speaking_time = 0.0

    # Helper containers
    times_acc = defaultdict(float)
    seg_counts = defaultdict(int)
    texts = defaultdict(list)
    first_ts = {}
    last_ts = {}

    for seg in segments:
        speaker = seg.speaker_id
        if speaker is None:  # Skip silence
            continue
        duration = max(0.0, seg.end - seg.start)
        times_acc[speaker] += duration
        seg_counts[speaker] += 1
        texts[speaker].append(seg.text)
        total_speaking_time += duration
        if speaker not in first_ts:
            first_ts[speaker] = seg.start
        last_ts[speaker] = seg.end

    # Basic stopword list (lightweight)
    stopwords = set(
        [
            'the','and','for','that','this','with','from','have','your','about','there','their','will','would',
            'what','when','where','which','while','shall','should','could','into','then','than','also','they',
            'them','you','are','was','were','been','being','can','our','out','any','but','not','just','like',
            'over','after','before','because','those','these','here','such','upon','only','each','other','more'
        ]
    )

    summary: dict = {}
    for speaker, total_time in times_acc.items():
        full_text = ' '.join(texts[speaker])
        # Tokenize words
        tokens = [
            w.lower()
            for w in re.findall(r"[A-Za-z']+", full_text)
            if len(w) > 3 and w.lower() not in stopwords
        ]
        word_count = len(re.findall(r"\w+", full_text))
        keyword_counts = Counter(tokens)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(8)]
        avg_words_per_segment = word_count / seg_counts[speaker] if seg_counts[speaker] else 0.0
        speaking_share = (total_time / total_speaking_time * 100.0) if total_speaking_time else 0.0
        excerpt = full_text[:300].strip()
        summary[speaker] = {
            'total_time': total_time,
            'segment_count': seg_counts[speaker],
            'word_count': word_count,
            'avg_words_per_segment': avg_words_per_segment,
            'speaking_share_pct': speaking_share,
            'first_ts': first_ts.get(speaker, 0.0),
            'last_ts': last_ts.get(speaker, 0.0),
            'top_keywords': top_keywords,
            'sample_excerpt': excerpt,
        }

    return summary


def write_speaker_summary(summary: dict, output_path: str) -> None:
    """Persist the speaker summary dictionary to a human-readable text file."""
    lines: list[str] = []
    lines.append('SPEAKER SUMMARY\n')
    if not summary:
        lines.append('No speaker data available.')
    else:
        # Order by total speaking time descending
        ordered = sorted(summary.items(), key=lambda kv: kv[1]['total_time'], reverse=True)
        for speaker, data in ordered:
            lines.append(f"=== Speaker: {speaker} ===")
            lines.append(f"Total Speaking Time: {data['total_time']:.2f} s")
            lines.append(f"Speaking Share: {data['speaking_share_pct']:.2f}%")
            lines.append(f"Segments: {data['segment_count']}")
            lines.append(f"Word Count: {data['word_count']}")
            lines.append(f"Avg Words / Segment: {data['avg_words_per_segment']:.2f}")
            lines.append(f"First Appearance: {data['first_ts']:.2f} s")
            lines.append(f"Last Appearance: {data['last_ts']:.2f} s")
            if data['top_keywords']:
                lines.append(f"Top Keywords: {', '.join(data['top_keywords'])}")
            if data['sample_excerpt']:
                lines.append('Sample Excerpt:')
                lines.append(data['sample_excerpt'])
            lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    try:
        os.startfile(output_path)
    except Exception:
        pass


def __diarize_audio(audio_path: str, num_speakers: int) -> list[int]:
    """
    Perform enhanced speaker diarization with improved audio preprocessing.
    Returns a list of speaker IDs for fixed-duration segments with better accuracy.
    """
    from pyAudioAnalysis.audioSegmentation import speaker_diarization

    print('Performing enhanced speaker diarization...')
    
    # Enhanced audio preprocessing for better diarization
    processed_audio_path = __preprocess_audio_for_diarization(audio_path)
    
    # Detect silence periods (optimized)
    silence_periods = __detect_silence(processed_audio_path if processed_audio_path else audio_path)

    # Speaker diarization with preprocessed audio
    audio_for_diarization = processed_audio_path if processed_audio_path else audio_path
    [flags, classes, accuracy] = speaker_diarization(audio_for_diarization, num_speakers, plot_res=False)
    flags = [int(flag) for flag in flags]

    # Enhanced flag adjustment with consistency improvements
    adjusted_flags = __adjust_flags_for_silence_enhanced(flags, silence_periods)
    
    # Clean up temporary processed audio file
    if processed_audio_path and processed_audio_path != audio_path:
        import os
        try:
            os.remove(processed_audio_path)
        except OSError:
            pass  # Ignore cleanup errors
    
    print(f'Enhanced speaker diarization completed with {len(set(adjusted_flags))} unique speakers detected')
    return adjusted_flags


def __detect_silence(audio_path: str, smoothing_filter_size: int = 50) -> list[tuple[float, float]]:
    """
    Detects silence periods in an audio file.

    :param audio_path: Path to the audio file.
    :param smoothing_filter_size: Size of the smoothing filter applied to energy signal.
    :return: A list of tuples representing silent periods (start_time, end_time).
    """
    from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
    from pyAudioAnalysis.ShortTermFeatures import feature_extraction

    # Extract short-term features
    [fs, x] = read_audio_file(audio_path)
    x = stereo_to_mono(x)  # Convert to mono if stereo

    # Calculate frame length and step size in samples
    frame_length_samples = int(0.050 * fs)  # 50 ms frame
    frame_step_samples = int(0.025 * fs)  # 25 ms step

    features, f_names = feature_extraction(x, fs, frame_length_samples, frame_step_samples)

    # Find the index of the energy feature
    energy_index = f_names.index('energy')
    energy = features[energy_index, :]

    # Smooth the energy signal
    if smoothing_filter_size > 1:
        energy = np.convolve(energy, np.ones(smoothing_filter_size) / smoothing_filter_size, mode='same')

    # Using the energy signal calculated above
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # Example heuristic: set threshold as mean minus half standard deviation
    energy_threshold = max(mean_energy - 1.2 * std_energy, 0.001)  # Ensure it doesn't go negative

    # Identify frames below the energy threshold
    silent_frames = energy < energy_threshold

    # Group silent frames into continuous silent periods
    silent_periods = []
    start_time = None
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start_time is None:
            start_time = i * 0.025  # Start time of the silent period
        elif not is_silent and start_time is not None:
            end_time = i * 0.025  # End time of the silent period
            silent_periods.append((start_time, end_time))
            start_time = None
    # Handle case where the last frame is silent
    if start_time is not None:
        silent_periods.append((start_time, len(silent_frames) * 0.025))

    return silent_periods


def __adjust_flags_for_silence(flags: list[int], silence_periods: list[tuple[float, float]]) -> list[int]:
    """
    Insert silence periods into the speaker flags array. This function adjusts the speaker flags array based on the identified silence periods. The speaker flags array is a list of speaker IDs for fixed-duration segments. The silence periods are represented by a speaker ID of -1.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param silence_periods: List of tuples representing silent periods (start_time, end_time)
    :return: Adjusted list of speaker IDs with silence periods inserted
    """
    # Adjust the flags array based on identified silence periods
    adjusted_flags = flags.copy()

    for silence_start, silence_end in silence_periods:
        start_index = int(silence_start / AS_SEGMENT_DURATION)
        end_index = int(silence_end / AS_SEGMENT_DURATION) + 1
        adjusted_flags[start_index:end_index] = [-1] * (end_index - start_index)

    return adjusted_flags


def __adjust_flags_for_silence_enhanced(flags: list[int], silence_periods: list[tuple[float, float]]) -> list[int]:
    """
    Enhanced version of silence adjustment with smoothing and consistency checks.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param silence_periods: List of tuples representing silent periods (start_time, end_time)
    :return: Enhanced adjusted list of speaker IDs with improved consistency
    """
    # Start with basic silence adjustment
    adjusted_flags = __adjust_flags_for_silence(flags, silence_periods)
    
    # Apply smoothing to reduce noise in speaker assignments
    smoothed_flags = __smooth_speaker_flags(adjusted_flags)
    
    return smoothed_flags


def __smooth_speaker_flags(flags: list[int], window_size: int = 3) -> list[int]:
    """
    Apply smoothing filter to speaker flags to reduce noise and inconsistencies.
    :param flags: List of speaker IDs
    :param window_size: Size of smoothing window
    :return: Smoothed speaker flags
    """
    if len(flags) < window_size:
        return flags
    
    smoothed = flags.copy()
    
    for i in range(window_size // 2, len(flags) - window_size // 2):
        window = flags[i - window_size // 2:i + window_size // 2 + 1]
        
        # Filter out silence (-1) for majority vote
        non_silence = [f for f in window if f != -1]
        
        if non_silence:
            # Use majority vote for non-silence segments
            from collections import Counter
            most_common = Counter(non_silence).most_common(1)[0][0]
            
            # Only apply smoothing if current segment is very short or isolated
            if flags[i] != -1 and (i == 0 or i == len(flags) - 1 or 
                                   flags[i-1] != flags[i] or flags[i+1] != flags[i]):
                smoothed[i] = most_common
    
    return smoothed


def __preprocess_audio_for_diarization(audio_path: str) -> str:
    """
    Preprocess audio file for better diarization accuracy.
    Applies noise reduction and normalization.
    :param audio_path: Path to original audio file
    :return: Path to processed audio file (may be same as input if processing fails)
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        import tempfile
        import os
        
        # Load audio with librosa for processing
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        # Apply noise reduction using spectral gating
        # Simple implementation - can be enhanced with more sophisticated methods
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Estimate noise floor from the first 0.5 seconds
        noise_frames = int(0.5 * sample_rate / 512)  # Assuming 512 hop length
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral gating (simple noise reduction)
        noise_gate_factor = 2.0
        mask = magnitude > (noise_gate_factor * noise_profile)
        cleaned_stft = stft * mask
        
        # Convert back to time domain
        cleaned_audio = librosa.istft(cleaned_stft)
        
        # Normalize audio
        cleaned_audio = librosa.util.normalize(cleaned_audio)
        
        # Save processed audio to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        sf.write(temp_path, cleaned_audio, sample_rate)
        
        print(f"Audio preprocessed and saved to: {temp_path}")
        return temp_path
        
    except ImportError:
        print("Warning: librosa or soundfile not available. Skipping audio preprocessing.")
        return audio_path
    except Exception as e:
        print(f"Warning: Audio preprocessing failed: {e}. Using original audio.")
        return audio_path


def __split_text_into_sentences(text: str) -> list[str]:
    """
    Simple function to split text into sentences based on punctuation.
    This is a naive implementation and can be replaced with more sophisticated NLP tools.
    """
    import re

    # TODO replace with a more sophisticated NLP tool?
    sentences = re.split(r'[.!?]\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def __get_segment_flags(flags: list[int], start_time: float, end_time: float) -> list[int]:
    """
    Get speaker flags for a segment based on the start and end times. Silences (represented by -1) are filtered out.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param start_time: Start time of the segment
    :param end_time: End time of the segment
    :return: List of speaker IDs for the segment
    """
    start_index = int(start_time / AS_SEGMENT_DURATION)
    end_index = int(end_time / AS_SEGMENT_DURATION) + 1
    segment_flags = flags[start_index:end_index]

    # Filter out pause/silence speaker IDs if defined (e.g., ID -1)
    return [flag for flag in segment_flags if flag != -1]


def __get_segment_flags_weighted(flags: list[int], start_time: float, end_time: float) -> dict[int, float]:
    """
    Get weighted speaker flags based on overlap calculation for better accuracy.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param start_time: Start time of the segment
    :param end_time: End time of the segment
    :return: Dictionary mapping speaker IDs to their weighted presence
    """
    if end_time <= start_time:
        return {}
    
    speaker_weights = {}
    segment_duration = end_time - start_time
    
    # Calculate precise overlap for each diarization segment
    current_time = start_time
    while current_time < end_time:
        next_time = min(current_time + AS_SEGMENT_DURATION, end_time)
        overlap_duration = next_time - current_time
        
        flag_index = int(current_time / AS_SEGMENT_DURATION)
        if flag_index < len(flags) and flags[flag_index] != -1:  # Not silence
            speaker_id = flags[flag_index]
            weight = overlap_duration / segment_duration
            speaker_weights[speaker_id] = speaker_weights.get(speaker_id, 0) + weight
        
        current_time = next_time
    
    return speaker_weights


def __assign_speaker_with_confidence(flags: list[int], start_time: float, end_time: float) -> int:
    """
    Assign speaker with confidence scoring using multiple methods.
    :param flags: List of speaker IDs from diarization
    :param start_time: Start time of the segment
    :param end_time: End time of the segment
    :return: Most confident speaker ID
    """
    # Method 1: Weighted overlap
    weighted_flags = __get_segment_flags_weighted(flags, start_time, end_time)
    if not weighted_flags:
        return None
    
    # Method 2: Simple majority in the middle 60% of the segment (more reliable)
    middle_start = start_time + 0.2 * (end_time - start_time)
    middle_end = end_time - 0.2 * (end_time - start_time)
    middle_flags = __get_segment_flags(flags, middle_start, middle_end)
    
    # Combine methods with confidence weighting
    if middle_flags:
        middle_speaker = max(set(middle_flags), key=middle_flags.count)
        # If middle segment agrees with weighted result, high confidence
        if middle_speaker in weighted_flags and weighted_flags[middle_speaker] > 0.3:
            return middle_speaker
    
    # Fall back to weighted result
    return max(weighted_flags.keys(), key=lambda k: weighted_flags[k])


def __map_sentences_to_speakers(transcription_segments: list[Segment], flags: list[int]) -> list[Segment]:
    """
    Map sentences to speakers based on the speaker diarization with enhanced alignment.
    This function uses improved overlap calculation and weighted majority voting for better accuracy.
    :param transcription_segments: List of Segments with 'start', 'end', 'text'
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :return: A list of Segments for each sentence with improved speaker assignments
    """
    result_segments: list[Segment] = []
    for segment in transcription_segments:
        segment_flags = __get_segment_flags_weighted(flags, segment.start, segment.end)

        if not segment_flags or len(set(segment_flags.keys())) == 1:
            # Segment has a unanimous speaker (or silence), keep it as is
            speaker_id = max(segment_flags.keys(), key=lambda k: segment_flags[k]) if segment_flags else None
            segment.speaker_id = speaker_id
            result_segments.append(segment)
        else:
            # Segment has mixed speakers, split into sentences with improved timing
            sentences = __split_text_into_sentences(segment.text)
            
            if not sentences:  # Handle empty text
                segment.speaker_id = max(segment_flags.keys(), key=lambda k: segment_flags[k]) if segment_flags else None
                result_segments.append(segment)
                continue
            
            # Calculate cumulative sentence positions for better timing estimation
            cumulative_chars = []
            total_chars = 0
            for sentence in sentences:
                total_chars += len(sentence)
                cumulative_chars.append(total_chars)
            
            total_duration = segment.end - segment.start
            
            for i, sentence in enumerate(sentences):
                # Improved timing calculation based on cumulative character positions
                if i == 0:
                    sentence_start_time = segment.start
                else:
                    sentence_start_time = segment.start + (cumulative_chars[i-1] / total_chars) * total_duration
                
                sentence_end_time = segment.start + (cumulative_chars[i] / total_chars) * total_duration
                
                # Use weighted speaker assignment with confidence scoring
                sentence_speaker = __assign_speaker_with_confidence(flags, sentence_start_time, sentence_end_time)
                
                result_segments.append(
                    Segment(
                        start=sentence_start_time,
                        end=sentence_end_time,
                        text=sentence.strip(),
                        speaker_id=sentence_speaker,
                    )
                )

    return result_segments


def __organize_by_speaker(transcription_segments: list[Segment]) -> list[Segment]:
    """
    Organize the transcription segments by speaker with consistency validation.
    This function groups consecutive segments by the same speaker and applies consistency checks.
    :param transcription_segments: List of Segments
    :return: List of Segments organized by speaker with validated assignments
    """
    if not transcription_segments:
        return []
    
    # Apply speaker consistency validation before organizing
    validated_segments = __validate_speaker_consistency(transcription_segments)
    
    organized_segments: list[Segment] = []
    current_speaker: int | None = None
    current_segment: Segment = None  # type: ignore

    for segment in validated_segments:
        # Check if the current segment is continuing
        if segment.speaker_id == current_speaker:
            # Append text to the current segment
            current_segment.text += ' ' + segment.text
            current_segment.end = segment.end
        else:
            # Finish the current segment and start a new one
            if current_segment:
                organized_segments.append(current_segment)

            current_speaker = segment.speaker_id
            # Make a copy of the segment to avoid modifying the original segment
            current_segment = Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=segment.speaker_id,
            )

    # Don't forget to add the last segment
    if current_segment:
        organized_segments.append(current_segment)

    return organized_segments


def __validate_speaker_consistency(segments: list[Segment]) -> list[Segment]:
    """
    Validate and correct speaker assignments for consistency.
    Uses pattern recognition to identify and fix likely speaker assignment errors.
    :param segments: List of segments with potentially inconsistent speaker assignments
    :return: List of segments with corrected speaker assignments
    """
    if len(segments) < 3:
        return segments
    
    corrected_segments = segments.copy()
    
    # Pattern 1: Fix isolated single segments (A-B-A pattern likely should be A-A-A)
    for i in range(1, len(corrected_segments) - 1):
        prev_speaker = corrected_segments[i-1].speaker_id
        curr_speaker = corrected_segments[i].speaker_id
        next_speaker = corrected_segments[i+1].speaker_id
        
        # If current segment is isolated and short, likely misassigned
        if (prev_speaker == next_speaker and 
            curr_speaker != prev_speaker and
            corrected_segments[i].end - corrected_segments[i].start < 2.0):  # Less than 2 seconds
            corrected_segments[i].speaker_id = prev_speaker
    
    # Pattern 2: Fix rapid alternations (likely same speaker with diarization noise)
    window_size = 3
    for i in range(len(corrected_segments) - window_size + 1):
        window = corrected_segments[i:i + window_size]
        speakers = [seg.speaker_id for seg in window]
        
        # If we have rapid alternation, use majority vote in local window
        if len(set(speakers)) > 1:
            durations = [seg.end - seg.start for seg in window]
            total_duration = sum(durations)
            
            # Weight by duration for majority vote
            speaker_durations = {}
            for seg in window:
                duration = seg.end - seg.start
                speaker_durations[seg.speaker_id] = speaker_durations.get(seg.speaker_id, 0) + duration
            
            majority_speaker = max(speaker_durations.keys(), key=lambda k: speaker_durations[k])
            
            # Apply correction if majority is strong (>60% of window duration)
            if speaker_durations[majority_speaker] / total_duration > 0.6:
                for j, seg in enumerate(window):
                    if seg.end - seg.start < 1.0:  # Only correct very short segments
                        corrected_segments[i + j].speaker_id = majority_speaker
    
    return corrected_segments


if __name__ == '__main__':
    media_path = input('Enter the path to the audio/video file: ')
    if not os.path.exists(media_path):
        raise FileNotFoundError(f'File not found: {media_path}')

    num_speakers = int(input('Enter the number of speakers: '))
    language = input('Enter the language of the audio file: ')

    transcription_segments = transcribe_media(media_path, num_speakers, language)

    # Initial transcript with numeric speaker IDs
    write_transcript(transcription_segments, TRANSCRIPT_PATH)

    unique_speakers = list(
        sorted(set(segment.speaker_id for segment in transcription_segments if segment.speaker_id is not None))
    )
    speaker_name_mapping = {}
    for speaker in unique_speakers:
        name = input(f'Enter name for speaker {speaker}: ')
        speaker_name_mapping[speaker] = name

    for segment in transcription_segments:
        if segment.speaker_id is not None:
            segment.speaker_id = speaker_name_mapping[segment.speaker_id]

    # Final transcript with speaker names
    write_transcript(transcription_segments, TRANSCRIPT_PATH)

    # Generate and write speaker summary (uses named speakers)
    speaker_summary = generate_speaker_summary(transcription_segments)
    write_speaker_summary(speaker_summary, SPEAKER_SUMMARY_PATH)

    # LLM Analysis Integration
    print("\n" + "="*50)
    print("AI LLM-POWERED MEETING ANALYSIS")
    print("="*50)
    
    llm_choice = input("Would you like to perform LLM analysis? (y/n): ").lower().strip()
    
    if llm_choice in ['y', 'yes']:
        try:
            from llm_analysis import analyze_meeting_files
            
            # Ask user for LLM provider preference
            print("\nAvailable LLM providers:")
            print("1. OpenAI (gpt-3.5-turbo) - Requires API key")
            print("2. Ollama (llama3) - Local, no API key needed")
            print("3. Anthropic (claude-3-sonnet) - Requires API key")
            print("4. Groq (meta-llama/llama-4-scout-17b-16e-instruct) - Fast inference, requires API key")
            
            provider_choice = input("Choose provider (1/2/3/4) or press Enter for OpenAI: ").strip()
            
            if provider_choice == "2":
                provider = "ollama"
                model = "llama3"
                print(">> Using Ollama (local). Make sure Ollama is running with llama3 model.")
            elif provider_choice == "3":
                provider = "anthropic"
                model = "claude-3-sonnet-20240229"
                print(">> Using Anthropic Claude. Make sure ANTHROPIC_API_KEY is set.")
            elif provider_choice == "4":
                provider = "groq"
                model = "meta-llama/llama-4-scout-17b-16e-instruct"
                print(">> Using Groq with Llama 4 Scout. Make sure GROQ_API_KEY is set.")
            else:
                provider = "openai"
                model = "gpt-3.5-turbo"
                print(">> Using OpenAI GPT. Make sure OPENAI_API_KEY is set.")
            
            print(f"\n>> Starting analysis with {provider}...")
            
            # Run LLM analysis
            analysis = analyze_meeting_files(
                transcript_path=TRANSCRIPT_PATH,
                speaker_summary_path=SPEAKER_SUMMARY_PATH,
                provider=provider,
                model=model
            )
            
            print("\n[SUCCESS] LLM Analysis completed successfully!")
            print("📄 Results saved to 'meeting_analysis.txt'")
            print("\n>> Quick Summary:")
            print(f"   • {len(analysis.key_discussion_points)} key discussion points identified")
            print(f"   • {len(analysis.decisions_made)} decisions documented")
            print(f"   • {len(analysis.action_items)} action items extracted")
            print(f"   • {len(analysis.follow_up_items)} follow-up items noted")
            print(f"   • Meeting effectiveness: {analysis.meeting_effectiveness_score}/10")
            
        except ImportError as e:
            print(f"[ERROR] LLM analysis dependencies not installed: {e}")
            print("💡 To install: pip install openai anthropic ollama")
        except Exception as e:
            print(f"[ERROR] Error during LLM analysis: {e}")
            print("💡 Check your API keys and network connection")
    
    print("\n" + "="*50)
    print("[SUCCESS] PROCESSING COMPLETE")
    print("="*50)
    print(f">> Transcript: {TRANSCRIPT_PATH}")
    print(f">> Speaker Summary: {SPEAKER_SUMMARY_PATH}")
    if llm_choice in ['y', 'yes']:
        print(f">> LLM Analysis: meeting_analysis.txt")
    print(f">> Run analysis_proj.py for metrics table")

#C:\Users\uzair\Desktop\AI-DL_Project\Meeting-Summarizer\meetenv\Scripts\python.exe C:\Users\uzair\Desktop\AI-DL_Project\Meeting-Summarizer\main.py 