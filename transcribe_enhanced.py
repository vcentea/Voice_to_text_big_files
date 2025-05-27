#!/usr/bin/env python3
"""
Enhanced Whisper Transcription with Speaker Diarization
Simple, reliable implementation focused on quality transcription.
Features:
- GPU acceleration with automatic CPU fallback
- Speaker diarization with pyannote.audio
- Clean SRT output format
- Progress monitoring
"""

import whisper
import torch
import sys
import os
import warnings
import time
import tempfile
import subprocess
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

def setup_environment():
    """Setup CUDA environment and suppress warnings"""
    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    
    # Add CUDA to PATH
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin"
    current_path = os.environ.get("PATH", "")
    if cuda_bin not in current_path:
        os.environ["PATH"] = f"{cuda_bin};{current_path}"
    
    # Suppress common warnings
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
    
    # Enable CUDA optimizations if available
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

def convert_audio_if_needed(audio_file):
    """Convert audio to WAV format if needed"""
    if audio_file.lower().endswith('.wav'):
        return audio_file
    
    print(f"🔄 Converting {Path(audio_file).suffix} to WAV...")
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

def perform_speaker_diarization(audio_file):
    """Perform speaker diarization using pyannote.audio"""
    try:
        from pyannote.audio import Pipeline
        print("\n👥 SPEAKER DIARIZATION STARTING...")
        print("=" * 50)
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found - speaker diarization will be disabled")
            return None
        
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📥 Loading speaker diarization model on {device.upper()}...")
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            if device == "cuda":
                pipeline = pipeline.to(torch.device("cuda"))
                
        except Exception as model_error:
            print(f"❌ Failed to load speaker diarization model: {model_error}")
            print("💡 Please accept the license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return None
        
        print("🔍 Analyzing speakers...")
        diarization = pipeline(audio_file)
        
        # Convert to speaker segments
        speaker_segments = []
        speakers_found = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            speakers_found.add(speaker)
        
        elapsed = time.time() - start_time
        print(f"✅ SPEAKER DIARIZATION COMPLETE!")
        print(f"👥 Found {len(speakers_found)} speakers: {', '.join(sorted(speakers_found))}")
        print(f"⏱️ Diarization time: {elapsed:.1f}s")
        print("=" * 50)
        
        return speaker_segments
        
    except ImportError:
        print("⚠️ pyannote.audio not available - continuing without speaker diarization")
        return None
    except Exception as e:
        print(f"⚠️ Speaker diarization failed: {e}")
        return None

def assign_speaker_to_segment(segment_start, segment_end, speaker_segments):
    """Assign speaker to a segment based on overlap"""
    if not speaker_segments:
        return "SPEAKER_UNKNOWN"
    
    segment_mid = (segment_start + segment_end) / 2
    
    for spk_seg in speaker_segments:
        if spk_seg['start'] <= segment_mid <= spk_seg['end']:
            return spk_seg['speaker']
    
    return "SPEAKER_UNKNOWN"

def format_timestamp_srt(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_audio(model, audio_file, output_file, speaker_segments=None, language="ro"):
    """Transcribe audio with Whisper and write SRT output"""
    print("\n🎙️ TRANSCRIPTION STARTING...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Transcribe with Whisper
    result = model.transcribe(
        audio_file,
        language=language,
        verbose=False,
        temperature=0.0,  # More deterministic
        fp16=torch.cuda.is_available()  # Use FP16 on GPU
    )
    
    print(f"✅ Transcription complete: {len(result['segments'])} segments")
    
    # Write SRT file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            start_ts = segment['start']
            end_ts = segment['end']
            text = segment['text'].strip()
            
            # Assign speaker
            speaker = assign_speaker_to_segment(start_ts, end_ts, speaker_segments)
            
            # Format timestamps
            start_formatted = format_timestamp_srt(start_ts)
            end_formatted = format_timestamp_srt(end_ts)
            
            # Format text with speaker
            if speaker != "SPEAKER_UNKNOWN":
                display_text = f"[{speaker}] {text}"
            else:
                display_text = text
            
            # Write SRT entry
            f.write(f"{i}\n")
            f.write(f"{start_formatted} --> {end_formatted}\n")
            f.write(f"{display_text}\n\n")
    
    elapsed = time.time() - start_time
    total_duration = result['segments'][-1]['end'] if result['segments'] else 0
    speed_ratio = total_duration / elapsed if elapsed > 0 else 0
    
    print(f"✅ TRANSCRIPTION COMPLETE!")
    print(f"⏱️ Processing time: {elapsed:.1f}s")
    print(f"🚀 Speed ratio: {speed_ratio:.1f}x real-time")
    print("=" * 60)
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_enhanced.py <audio_file> [output_file] [--auto-lang]")
        print("Example: python transcribe_enhanced.py audio.wav output.srt")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output.srt"
    auto_lang = "--auto-lang" in sys.argv
    
    print("🚀 Enhanced Whisper Transcription + Speaker Diarization")
    print("💎 Simple, Reliable, High-Quality")
    print("=" * 60)
    
    setup_environment()
    setup_ffmpeg()
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN available: {hf_token[:10]}...")
    else:
        print("⚠️ HF_TOKEN not found - speaker diarization will be disabled")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Check GPU/CPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Using GPU: {gpu_name}")
        print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    else:
        device = "cpu"
        print("🔧 Using CPU (GPU not available)")
    
    print(f"📁 Input file: {audio_file}")
    print(f"📁 Output file: {output_file}")
    
    # Load Whisper model
    print("\n📥 Loading Whisper large-v3 model...")
    try:
        model = whisper.load_model("large-v3", device=device)
        print(f"✅ Model loaded successfully on {device.upper()}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Convert audio if needed
    working_audio = convert_audio_if_needed(audio_file)
    
    # Language detection or preset
    if auto_lang:
        print("🌍 Auto-detecting language...")
        language = None  # Let Whisper detect
    else:
        language = "ro"  # Default to Romanian
        print(f"🌍 Using language: {language}")
    
    # Perform speaker diarization
    speaker_segments = perform_speaker_diarization(working_audio)
    
    # Transcribe audio
    result = transcribe_audio(model, working_audio, output_file, speaker_segments, language)
    
    # Clean up temporary files
    if working_audio != audio_file and os.path.exists(working_audio):
        try:
            os.unlink(working_audio)
            print("🗑️ Temporary audio file cleaned up")
        except:
            pass
    
    # Final summary
    print("=" * 60)
    print("🎉 ENHANCED TRANSCRIPTION COMPLETED!")
    print("=" * 60)
    
    total_duration = result['segments'][-1]['end'] if result['segments'] else 0
    unique_speakers = len(set(
        assign_speaker_to_segment(seg['start'], seg['end'], speaker_segments) 
        for seg in result['segments'] 
        if speaker_segments
    )) if speaker_segments else 1
    
    print(f"📊 SUMMARY:")
    print(f"   📁 Output file: {output_file}")
    print(f"   🌍 Language: {result.get('language', 'auto-detected')}")
    print(f"   ⏱️  Audio duration: {total_duration:.1f} seconds")
    print(f"   📝 Total segments: {len(result['segments'])}")
    print(f"   👥 Speakers detected: {unique_speakers}")
    print("💎 Simple, reliable transcription complete!")

if __name__ == "__main__":
    main() 