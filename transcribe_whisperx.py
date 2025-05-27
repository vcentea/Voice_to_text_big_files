#!/usr/bin/env python3
"""
HIGH-QUALITY WhisperX Transcription with Speaker Diarization
Features:
- WhisperX for superior accuracy and word-level timestamps
- Forced phoneme alignment for precise timing
- Advanced speaker diarization integration
- VAD preprocessing to reduce hallucinations
- Batched inference for speed
- Quality validation and error detection
"""

import whisperx
import torch
import sys
import os
import warnings
import time
import gc
import tempfile
import subprocess
from pathlib import Path

# Load environment variables from .env file
def load_environment_variables():
    """Load environment variables from .env file with proper error handling"""
    try:
        from dotenv import load_dotenv
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv()
            print("✅ Environment variables loaded from .env file")
            
            # Verify HF_TOKEN is loaded
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                print(f"✅ HF_TOKEN found in .env: {hf_token[:10]}...")
            else:
                print("⚠️ HF_TOKEN not found in .env file")
                print("💡 Please add your HuggingFace token to .env file:")
                print("   HF_TOKEN=your_token_here")
        else:
            print("⚠️ .env file not found")
            print("💡 Create .env file with your HuggingFace token:")
            print("   HF_TOKEN=your_token_here")
    except ImportError:
        print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    except Exception as e:
        print(f"⚠️ Could not load .env file: {e}")

# Load environment variables at startup
load_environment_variables()

# Constants for quality optimization
BATCH_SIZE_GPU = 16  # Optimal for RTX 3090 Ti
BATCH_SIZE_CPU = 4   # Conservative for CPU
MIN_SEGMENT_LENGTH = 0.5  # Minimum segment length to avoid zero-duration
MAX_REPETITION_RATIO = 0.3  # Maximum allowed repetition ratio
COMPUTE_TYPE_GPU = "float16"  # Optimal for modern GPUs
COMPUTE_TYPE_CPU = "int8"     # Efficient for CPU

def setup_environment():
    """Setup optimal environment for WhisperX"""
    # Set environment variables
    os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    
    # Add CUDA to PATH
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin"
    current_path = os.environ.get("PATH", "")
    if cuda_bin not in current_path:
        os.environ["PATH"] = f"{cuda_bin};{current_path}"
    
    # Suppress specific warnings but keep important ones
    warnings.filterwarnings("ignore", message=".*You have requested a HuggingFace token.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisperx")
    
    # Optimize CUDA if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("🚀 CUDA optimizations enabled")

def setup_ffmpeg():
    """Setup FFmpeg path"""
    ffmpeg_path = r"C:\Users\vladc\Desktop\shareX\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = f"{ffmpeg_dir};{current_path}"
            print(f"🔧 Added FFmpeg to PATH: {ffmpeg_dir}")

def get_device_config():
    """Get optimal device configuration"""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = COMPUTE_TYPE_GPU
        batch_size = BATCH_SIZE_GPU
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ Using GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✅ Compute Type: {compute_type}")
        print(f"✅ Batch Size: {batch_size}")
        
        return device, compute_type, batch_size
    else:
        device = "cpu"
        compute_type = COMPUTE_TYPE_CPU
        batch_size = BATCH_SIZE_CPU
        
        print(f"🔧 Using CPU (GPU not available)")
        print(f"✅ Compute Type: {compute_type}")
        print(f"✅ Batch Size: {batch_size}")
        
        return device, compute_type, batch_size

def convert_audio_if_needed(audio_file):
    """Convert audio to optimal format for WhisperX if needed"""
    if audio_file.lower().endswith(('.wav', '.flac')):
        return audio_file
    
    print(f"🔄 Converting {Path(audio_file).suffix} to WAV for optimal processing...")
    
    temp_wav = tempfile.mktemp(suffix='.wav')
    
    ffmpeg_cmd = [
        "ffmpeg", "-i", audio_file,
        "-acodec", "pcm_s16le",
        "-ar", "16000", 
        "-ac", "1",
        "-y", temp_wav
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ Audio converted to: {temp_wav}")
            return temp_wav
        else:
            print(f"⚠️ Audio conversion failed: {result.stderr}")
            return audio_file
    except Exception as e:
        print(f"❌ Audio conversion error: {e}")
        return audio_file

def detect_repetitions(segments):
    """Detect and count repetitive patterns in transcription"""
    if not segments:
        return 0.0, []
    
    repetitive_segments = []
    total_words = 0
    repetitive_words = 0
    
    for i, segment in enumerate(segments):
        text = segment.get('text', '').strip().lower()
        words = text.split()
        total_words += len(words)
        
        # Check for word repetitions within segment
        if len(words) > 2:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate repetition ratio for this segment
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            if len(words) > 0 and repeated_words / len(words) > 0.5:
                repetitive_segments.append(i)
                repetitive_words += repeated_words
        
        # Check for segment-to-segment repetition
        if i > 0:
            prev_text = segments[i-1].get('text', '').strip().lower()
            if prev_text and text and len(text) > 10:
                # Check if segments are too similar
                prev_words = set(prev_text.split())
                curr_words = set(text.split())
                if len(prev_words) > 0:
                    similarity = len(prev_words & curr_words) / len(prev_words)
                    if similarity > 0.8:
                        repetitive_segments.append(i)
    
    repetition_ratio = repetitive_words / total_words if total_words > 0 else 0.0
    return repetition_ratio, repetitive_segments

def clean_repetitive_segments(segments):
    """Clean repetitive segments from transcription"""
    if not segments:
        return segments
    
    repetition_ratio, repetitive_indices = detect_repetitions(segments)
    
    if repetition_ratio > MAX_REPETITION_RATIO:
        print(f"⚠️ High repetition detected: {repetition_ratio:.1%}")
        print(f"🔧 Cleaning {len(repetitive_indices)} repetitive segments...")
        
        # Remove repetitive segments
        cleaned_segments = []
        for i, segment in enumerate(segments):
            if i not in repetitive_indices:
                cleaned_segments.append(segment)
            else:
                print(f"   Removed segment {i}: {segment.get('text', '')[:50]}...")
        
        print(f"✅ Cleaned transcription: {len(cleaned_segments)}/{len(segments)} segments kept")
        return cleaned_segments
    
    return segments

def transcribe_with_whisperx(audio_file, device, compute_type, batch_size, language=None):
    """Transcribe audio using WhisperX with quality validation"""
    print("\n🎙️ WHISPERX TRANSCRIPTION STARTING...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Load WhisperX model
        print(f"📥 Loading WhisperX large-v3 model on {device.upper()}...")
        model = whisperx.load_model(
            "large-v3", 
            device, 
            compute_type=compute_type,
            asr_options={
                "suppress_numerals": True,
                "max_new_tokens": None,
                "clip_timestamps": "0",
                "hallucination_silence_threshold": None,
            }
        )
        print(f"✅ WhisperX model loaded")
        
        # 2. Load audio
        print(f"📥 Loading audio: {audio_file}")
        audio = whisperx.load_audio(audio_file)
        audio_duration = len(audio) / 16000
        print(f"✅ Audio loaded: {audio_duration:.1f} seconds")
        
        # 3. Transcribe with WhisperX (includes VAD)
        print(f"🚀 Transcribing with batch size {batch_size}...")
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language,
            print_progress=True,
            combined_progress=True
        )
        
        print(f"✅ Initial transcription complete: {len(result['segments'])} segments")
        
        # Clean up model memory
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # 4. Quality validation
        repetition_ratio, repetitive_indices = detect_repetitions(result['segments'])
        print(f"📊 Quality check - Repetition ratio: {repetition_ratio:.1%}")
        
        if repetition_ratio > MAX_REPETITION_RATIO:
            print(f"⚠️ Quality issue detected! Cleaning repetitive segments...")
            result['segments'] = clean_repetitive_segments(result['segments'])
        
        # 5. Forced alignment for precise timestamps
        if result['segments']:
            print(f"🎯 Loading alignment model for {result.get('language', 'detected language')}...")
            
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=device
                )
                
                print(f"🔍 Performing forced alignment...")
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    device, 
                    return_char_alignments=False
                )
                
                print(f"✅ Forced alignment complete")
                
                # Clean up alignment model
                del model_a
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️ Alignment failed: {e}")
                print("🔧 Continuing with original timestamps...")
        
        transcription_time = time.time() - start_time
        speed_ratio = audio_duration / transcription_time if transcription_time > 0 else 0
        
        print(f"\n✅ WHISPERX TRANSCRIPTION COMPLETE!")
        print(f"⏱️ Processing time: {transcription_time:.1f}s")
        print(f"🚀 Speed ratio: {speed_ratio:.1f}x real-time")
        print("=" * 60)
        
        return result, transcription_time, speed_ratio
        
    except Exception as e:
        print(f"❌ WhisperX transcription failed: {e}")
        raise

def perform_speaker_diarization_whisperx(audio_file, device):
    """Perform speaker diarization optimized for WhisperX"""
    try:
        print("\n👥 SPEAKER DIARIZATION STARTING...")
        print("=" * 50)
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found in .env file - speaker diarization disabled")
            print("💡 Add your HuggingFace token to .env file to enable speaker diarization")
            return None
        
        print(f"✅ Using HF_TOKEN from .env file: {hf_token[:10]}...")
        
        start_time = time.time()
        
        # Load diarization model
        print(f"📥 Loading speaker diarization model on {device.upper()}...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, 
            device=device
        )
        print(f"✅ Diarization model loaded")
        
        # Perform diarization
        print("🔍 Analyzing speakers...")
        diarize_segments = diarize_model(audio_file)
        
        # Process results
        speakers_found = set()
        for segment in diarize_segments.itertracks(yield_label=True):
            speakers_found.add(segment[2])
        
        elapsed = time.time() - start_time
        print(f"✅ SPEAKER DIARIZATION COMPLETE!")
        print(f"👥 Found {len(speakers_found)} speakers: {', '.join(sorted(speakers_found))}")
        print(f"⏱️ Diarization time: {elapsed:.1f}s")
        print("=" * 50)
        
        return diarize_segments
        
    except Exception as e:
        print(f"⚠️ Speaker diarization failed: {e}")
        print("🔧 Continuing without speaker labels...")
        return None

def assign_speakers_to_segments(result, diarize_segments):
    """Assign speaker labels to transcription segments"""
    if not diarize_segments or not result.get('segments'):
        return result
    
    print("🏷️ Assigning speaker labels to segments...")
    
    # Use WhisperX's built-in speaker assignment
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Count speakers
    speakers_in_transcript = set()
    for segment in result['segments']:
        if 'speaker' in segment:
            speakers_in_transcript.add(segment['speaker'])
    
    print(f"✅ Speaker assignment complete: {len(speakers_in_transcript)} speakers")
    return result

def format_timestamp_srt(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def write_enhanced_srt(result, output_file):
    """Write enhanced SRT file with speaker labels and quality validation"""
    print(f"📝 Writing enhanced SRT to: {output_file}")
    
    segments = result.get('segments', [])
    if not segments:
        print("❌ No segments to write")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        segment_count = 0
        
        for i, segment in enumerate(segments):
            start = segment.get('start', 0)
            end = segment.get('end', start + 1)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            
            # Skip empty or too short segments
            if not text or (end - start) < MIN_SEGMENT_LENGTH:
                continue
            
            segment_count += 1
            
            # Format timestamps
            start_time = format_timestamp_srt(start)
            end_time = format_timestamp_srt(end)
            
            # Format text with speaker
            speaker_text = f"[{speaker}] {text}" if speaker != 'SPEAKER_UNKNOWN' else text
            
            # Write SRT entry
            f.write(f"{segment_count}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{speaker_text}\n\n")
    
    print(f"✅ Enhanced SRT written: {segment_count} segments")

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_whisperx.py <audio_file> [output_file] [--language code]")
        print("Example: python transcribe_whisperx.py audio.wav output.srt --language ro")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "whisperx_output.srt"
    
    # Parse language option
    language = None
    if "--language" in sys.argv:
        lang_index = sys.argv.index("--language")
        if lang_index + 1 < len(sys.argv):
            language = sys.argv[lang_index + 1]
    
    print("🚀 HIGH-QUALITY WHISPERX Transcription + Speaker Diarization")
    print("🎯 Superior Accuracy with Forced Alignment")
    print("=" * 70)
    
    setup_environment()
    setup_ffmpeg()
    
    # Check HF token from .env file
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN loaded from .env: {hf_token[:10]}...")
        print("✅ Speaker diarization will be enabled")
    else:
        print("⚠️ HF_TOKEN not found in .env file")
        print("⚠️ Speaker diarization will be disabled")
        print("💡 To enable speaker diarization:")
        print("   1. Get token from: https://huggingface.co/settings/tokens")
        print("   2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. Add to .env file: HF_TOKEN=your_token_here")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Get device configuration
    device, compute_type, batch_size = get_device_config()
    
    print(f"📁 Input file: {audio_file}")
    print(f"📁 Output file: {output_file}")
    if language:
        print(f"🌍 Language: {language}")
    
    # Convert audio if needed
    working_audio = convert_audio_if_needed(audio_file)
    
    try:
        # Main transcription with WhisperX
        result, transcription_time, speed_ratio = transcribe_with_whisperx(
            working_audio, device, compute_type, batch_size, language
        )
        
        # Speaker diarization
        diarize_segments = perform_speaker_diarization_whisperx(working_audio, device)
        
        # Assign speakers to segments
        if diarize_segments:
            result = assign_speakers_to_segments(result, diarize_segments)
        
        # Write enhanced SRT
        write_enhanced_srt(result, output_file)
        
        # Clean up temporary audio file
        if working_audio != audio_file and os.path.exists(working_audio):
            try:
                os.unlink(working_audio)
                print("🗑️ Temporary audio file cleaned up")
            except:
                pass
        
        # Final summary
        print("=" * 70)
        print("🎉 HIGH-QUALITY TRANSCRIPTION COMPLETED!")
        print("=" * 70)
        
        total_duration = result['segments'][-1]['end'] if result['segments'] else 0
        unique_speakers = len(set(
            seg.get('speaker', 'UNKNOWN') 
            for seg in result['segments'] 
            if seg.get('speaker')
        ))
        
        print(f"📊 QUALITY SUMMARY:")
        print(f"   📁 Output file: {output_file}")
        print(f"   🌍 Language: {result.get('language', 'auto-detected')}")
        print(f"   ⏱️  Audio duration: {total_duration:.1f} seconds")
        print(f"   🚀 Processing time: {transcription_time:.1f} seconds")
        print(f"   ⚡ Speed ratio: {speed_ratio:.1f}x real-time")
        print(f"   📝 Total segments: {len(result['segments'])}")
        print(f"   👥 Speakers detected: {unique_speakers}")
        print("💎 WhisperX superior quality with forced alignment!")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 