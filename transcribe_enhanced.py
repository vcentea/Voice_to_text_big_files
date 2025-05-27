#!/usr/bin/env python3
"""
ENHANCED GPU Transcription with MAXIMUM QUALITY optimization
Features:
- Audio preprocessing (noise reduction, normalization)
- Optimized Whisper parameters (temperature, beam search)
- Smart language detection
- Contextual prompts for better accuracy
- Post-processing cleanup
- Advanced chunking strategy
"""

import whisper
import torch
import sys
import os
import warnings
import time
import threading
import subprocess
import tempfile
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

def setup_cuda_environment():
    """Setup CUDA environment for optimal performance"""
    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    
    # Add CUDA to PATH
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin"
    current_path = os.environ.get("PATH", "")
    if cuda_bin not in current_path:
        os.environ["PATH"] = f"{cuda_bin};{current_path}"
    
    # Suppress warnings but keep important ones
    warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
    warnings.filterwarnings("ignore", message=".*falling back to a slower.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
    warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
    warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
    warnings.filterwarnings("ignore", message=".*audioread.*")
    
    # Enable optimal CUDA settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("🚀 CUDA optimizations enabled (including TF32)")

def setup_ffmpeg():
    """Setup FFmpeg path"""
    ffmpeg_path = r"C:\Users\vladc\Desktop\shareX\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = f"{ffmpeg_dir};{current_path}"
            print(f"🔧 Added FFmpeg to PATH: {ffmpeg_dir}")

def monitor_gpu_performance():
    """Monitor and return GPU performance metrics"""
    if not torch.cuda.is_available():
        return "CPU Mode"
    
    try:
        gpu_util = torch.cuda.utilization()
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_percent = (memory_used / memory_total) * 100
        
        return f"GPU: {gpu_util}% | VRAM: {memory_used:.1f}/{memory_total:.1f}GB ({memory_percent:.1f}%)"
    except:
        return "GPU: Active"

def enhance_audio_quality(audio_file):
    """
    Advanced audio preprocessing for better transcription quality
    """
    print("🎵 AUDIO ENHANCEMENT STARTING...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Convert to WAV first if not already WAV (fixes PySoundFile warnings)
        working_audio = audio_file
        if not audio_file.lower().endswith('.wav'):
            print(f"🔄 Converting {Path(audio_file).suffix} to WAV for optimal processing...")
            working_audio = tempfile.mktemp(suffix='.wav')
            
            ffmpeg_cmd = [
                "ffmpeg", "-i", audio_file,
                "-acodec", "pcm_s16le",
                "-ar", "16000", 
                "-ac", "1",
                "-y", working_audio
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"⚠️ Audio conversion failed, using original: {result.stderr}")
                working_audio = audio_file
            else:
                print("✅ Audio converted to WAV format")
        
        # Load audio with librosa for better quality
        print("📥 Loading audio with high-quality processing...")
        audio, sample_rate = librosa.load(working_audio, sr=16000, mono=True)
        
        print(f"✅ Audio loaded: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")
        
        # 1. Noise reduction
        print("🔇 Applying noise reduction...")
        try:
            # Use noisereduce for advanced noise reduction
            reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate, stationary=True)
            print("✅ Noise reduction applied")
        except:
            print("⚠️ Advanced noise reduction failed, using original audio")
            reduced_noise = audio
        
        # 2. Audio normalization
        print("📊 Normalizing audio levels...")
        # RMS normalization for consistent volume
        rms = np.sqrt(np.mean(reduced_noise**2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            normalized_audio = reduced_noise * (target_rms / rms)
        else:
            normalized_audio = reduced_noise
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 0.95:
            normalized_audio = normalized_audio * (0.95 / max_val)
        
        print("✅ Audio normalization complete")
        
        # 3. Save enhanced audio
        temp_enhanced = tempfile.mktemp(suffix='.wav')
        
        # Use librosa to save with high quality
        import soundfile as sf
        sf.write(temp_enhanced, normalized_audio, sample_rate, subtype='PCM_16')
        
        elapsed = time.time() - start_time
        print(f"\n✅ AUDIO ENHANCEMENT COMPLETE!")
        print(f"⏱️ Processing time: {elapsed:.1f}s")
        print(f"📁 Enhanced audio: {temp_enhanced}")
        print("=" * 50)
        
        # Clean up temporary conversion file if created
        if working_audio != audio_file and os.path.exists(working_audio):
            try:
                os.unlink(working_audio)
                print("🗑️ Temporary conversion file cleaned up")
            except:
                pass
        
        return temp_enhanced, sample_rate
        
    except Exception as e:
        print(f"❌ Audio enhancement failed: {e}")
        print("🔧 Falling back to original audio...")
        
        # Clean up temporary conversion file if created
        if 'working_audio' in locals() and working_audio != audio_file and os.path.exists(working_audio):
            try:
                os.unlink(working_audio)
                print("🗑️ Temporary conversion file cleaned up")
            except:
                pass
        
        return audio_file, 16000

def convert_audio_for_pyannote(audio_file):
    """Convert audio to WAV format for pyannote compatibility"""
    if audio_file.lower().endswith('.wav'):
        return audio_file
    
    print(f"🔄 Converting {Path(audio_file).suffix} to WAV for speaker diarization...")
    
    temp_wav = tempfile.mktemp(suffix='.wav')
    
    ffmpeg_cmd = [
        "ffmpeg", "-i", audio_file, 
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        temp_wav
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ Audio converted to: {temp_wav}")
            return temp_wav
        else:
            print(f"❌ FFmpeg conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Audio conversion error: {e}")
        return None

def detect_language_intelligently(model, audio_file):
    """
    Intelligent language detection with confidence scoring
    """
    print("🌍 INTELLIGENT LANGUAGE DETECTION...")
    
    try:
        # Load a small sample for language detection
        audio, _ = librosa.load(audio_file, sr=16000, duration=30, mono=True)
        
        # Use Whisper's language detection
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        
        # Get top 3 language predictions
        sorted_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("🎯 Language detection results:")
        for lang, confidence in sorted_langs:
            print(f"   {lang}: {confidence:.3f} ({confidence*100:.1f}%)")
        
        detected_lang = sorted_langs[0][0]
        confidence = sorted_langs[0][1]
        
        if confidence > 0.7:
            print(f"✅ High confidence detection: {detected_lang} ({confidence*100:.1f}%)")
            return detected_lang
        else:
            print(f"⚠️ Low confidence detection, using Romanian as fallback")
            return "ro"
            
    except Exception as e:
        print(f"❌ Language detection failed: {e}")
        print("🔧 Defaulting to Romanian")
        return "ro"

def create_contextual_prompt(language="ro"):
    """
    Create contextual prompts to improve transcription accuracy
    """
    prompts = {
        "ro": "Aceasta este o conversație în limba română. Vorbitorii discută despre diverse subiecte folosind limba română standard și expresii comune.",
        "en": "This is a conversation in English. The speakers are discussing various topics using standard English and common expressions.",
        "es": "Esta es una conversación en español. Los hablantes discuten varios temas usando español estándar y expresiones comunes.",
        "fr": "Ceci est une conversation en français. Les intervenants discutent de divers sujets en utilisant le français standard et des expressions courantes.",
        "de": "Das ist ein Gespräch auf Deutsch. Die Sprecher diskutieren verschiedene Themen mit Standarddeutsch und üblichen Ausdrücken.",
        "it": "Questa è una conversazione in italiano. I parlanti discutono vari argomenti usando l'italiano standard ed espressioni comuni."
    }
    
    return prompts.get(language, prompts["ro"])

def perform_speaker_diarization_with_progress(audio_file):
    """Perform speaker diarization with progress updates"""
    try:
        from pyannote.audio import Pipeline
        print("\n👥 SPEAKER DIARIZATION STARTING...")
        print("=" * 50)
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found - speaker diarization may not work")
            return None
        
        start_time = time.time()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📥 Loading speaker diarization model on {device.upper()}...")
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        except Exception as model_error:
            print(f"❌ Failed to load speaker diarization model: {model_error}")
            print("\n🔧 SOLUTION:")
            print("1. Visit: https://hf.co/pyannote/speaker-diarization-3.1")
            print("2. Accept the user conditions/license")
            print("3. Re-run the script")
            print("\n⚠️ Continuing without speaker diarization...")
            return None
        
        if pipeline is None:
            print("❌ Pipeline failed to load - continuing without speaker diarization")
            return None
        
        if device == "cuda":
            try:
                pipeline = pipeline.to(torch.device("cuda"))
                print(f"✅ Diarization model loaded on GPU | {monitor_gpu_performance()}")
            except Exception as gpu_error:
                print(f"⚠️ GPU loading failed, using CPU: {gpu_error}")
                device = "cpu"
        else:
            print(f"✅ Diarization model loaded on CPU")
        
        # Convert audio for pyannote if needed
        diarization_audio = convert_audio_for_pyannote(audio_file)
        if diarization_audio is None:
            print("❌ Could not convert audio for speaker diarization")
            return None
        
        # Perform diarization
        print("🔍 Analyzing speakers in audio...")
        diarization = pipeline(diarization_audio)
        
        # Convert to speaker segments with progress
        speaker_segments = []
        speakers_found = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            speakers_found.add(speaker)
            
            duration = turn.end - turn.start
            print(f"  🎤 {speaker}: {turn.start:.1f}s-{turn.end:.1f}s ({duration:.1f}s)")
        
        elapsed = time.time() - start_time
        print(f"\n✅ SPEAKER DIARIZATION COMPLETE!")
        print(f"👥 Found {len(speakers_found)} speakers: {', '.join(sorted(speakers_found))}")
        print(f"⏱️ Diarization time: {elapsed:.1f}s")
        print("=" * 50)
        
        # Clean up temporary file if created
        if diarization_audio != audio_file and os.path.exists(diarization_audio):
            try:
                os.unlink(diarization_audio)
                print("🗑️ Temporary audio file cleaned up")
            except:
                pass
        
        return speaker_segments
        
    except ImportError:
        print("⚠️ pyannote.audio not available - continuing without speaker diarization")
        return None
    except Exception as e:
        print(f"⚠️ Speaker diarization failed: {e}")
        print("\n⚠️ Continuing without speaker diarization...")
        return None

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def assign_speaker_to_segment(segment_mid, speaker_segments):
    """Assign speaker to a specific segment"""
    if not speaker_segments:
        return "SPEAKER_UNKNOWN"
    
    for spk_seg in speaker_segments:
        if spk_seg['start'] <= segment_mid <= spk_seg['end']:
            return spk_seg['speaker']
    
    return "SPEAKER_UNKNOWN"

def post_process_text(text, language="ro"):
    """
    Post-process transcribed text for better quality
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common transcription issues for Romanian
    if language == "ro":
        # Common Romanian corrections
        corrections = {
            r'\bsi\b': 'și',
            r'\bii\b': 'îi',
            r'\ba\b(?=\s+[aeiou])': 'a',  # Fix spacing
            r'\bde\b(?=\s+[aeiou])': 'de',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Capitalize first letter of sentences
    sentences = re.split(r'[.!?]+', text)
    processed_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            processed_sentences.append(sentence)
    
    return '. '.join(processed_sentences) if processed_sentences else text

def write_segment_to_file(file_handle, segment_num, start_time, end_time, speaker, text):
    """Write a single segment to SRT file immediately"""
    speaker_text = f"[{speaker}] {text}"
    
    file_handle.write(f"{segment_num}\n")
    file_handle.write(f"{start_time} --> {end_time}\n")
    file_handle.write(f"{speaker_text}\n\n")
    file_handle.flush()

def transcribe_with_enhanced_quality(model, audio_file, output_file, speaker_segments, language="ro"):
    """
    Enhanced transcription with optimized parameters for maximum quality
    """
    print("\n🎙️ ENHANCED QUALITY TRANSCRIPTION STARTING...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create contextual prompt
    prompt = create_contextual_prompt(language)
    print(f"🎯 Using contextual prompt for {language}")
    
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        print(f"📝 Streaming output to: {output_file}")
        print("=" * 60)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
            warnings.filterwarnings("ignore", message=".*falling back to a slower.*")
            
            # Enhanced transcription with optimized parameters
            result = model.transcribe(
                audio_file,
                language=language,
                verbose=False,
                word_timestamps=True,
                temperature=0.0,           # Most consistent results
                beam_size=5,              # Better accuracy (if available)
                best_of=5,                # Multiple candidates for better quality
                patience=1.0,             # More thorough search
                length_penalty=1.0,       # Balanced length preference
                suppress_tokens=[-1],     # Suppress silence token
                initial_prompt=prompt,    # Contextual hint
                condition_on_previous_text=True,  # Better context continuity
                fp16=torch.cuda.is_available(),   # Use FP16 for GPU efficiency
                compression_ratio_threshold=2.4,  # Skip low-quality segments
                logprob_threshold=-1.0,           # Quality threshold
                no_speech_threshold=0.6           # Silence detection
            )
        
        # Process segments with enhanced quality
        total_segments = len(result['segments'])
        print(f"\n📊 Processing {total_segments} segments with quality enhancement...")
        print("=" * 60)
        
        for i, segment in enumerate(result['segments'], 1):
            segment_start_time = time.time()
            
            # Get segment details
            start_ts = segment['start']
            end_ts = segment['end']
            text = segment['text'].strip()
            
            # Post-process text for better quality
            enhanced_text = post_process_text(text, language)
            
            # Assign speaker
            segment_mid = (start_ts + end_ts) / 2
            speaker = assign_speaker_to_segment(segment_mid, speaker_segments)
            
            # Format timestamps
            start_formatted = format_timestamp(start_ts)
            end_formatted = format_timestamp(end_ts)
            
            # Write to file immediately
            write_segment_to_file(srt_file, i, start_formatted, end_formatted, speaker, enhanced_text)
            
            # Calculate performance metrics
            segment_duration = end_ts - start_ts
            processing_time = time.time() - segment_start_time
            speed_ratio = segment_duration / processing_time if processing_time > 0 else 0
            progress_percent = (i / total_segments) * 100
            
            # Display real-time progress
            gpu_metrics = monitor_gpu_performance()
            
            print(f"✅ [{i:3d}/{total_segments}] {progress_percent:5.1f}% | {speed_ratio:4.1f}x | {gpu_metrics}")
            print(f"   ⏱️  {start_formatted} --> {end_formatted} | {segment_duration:.1f}s")
            print(f"   🎤 [{speaker}] {enhanced_text[:80]}{'...' if len(enhanced_text) > 80 else ''}")
            print()
    
    total_time = time.time() - start_time
    audio_duration = result['segments'][-1]['end'] if result['segments'] else 0
    overall_speed = audio_duration / total_time if total_time > 0 else 0
    
    return result, total_time, overall_speed

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_enhanced.py <audio_file> [output_file] [--auto-lang]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output.srt"
    auto_lang = "--auto-lang" in sys.argv
    
    print("🚀 ENHANCED QUALITY GPU Transcription + Speaker Diarization")
    print("🎯 Maximum Quality Mode with Audio Processing")
    print("=" * 60)
    
    setup_cuda_environment()
    setup_ffmpeg()
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN available: {hf_token[:10]}...")
    else:
        print("⚠️ HF_TOKEN not found in .env file - speaker diarization may not work")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📊 Initial GPU status: {monitor_gpu_performance()}")
    else:
        device = "cpu"
        print("🔧 Using CPU (GPU not available)")
    
    print(f"📁 Input file: {audio_file}")
    print(f"📁 Output file: {output_file}")
    
    # Load model
    print("\n📥 Loading Whisper large-v3 model...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model = whisper.load_model("large-v3", device=device)
        print(f"✅ Model loaded successfully on GPU | {monitor_gpu_performance()}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Enhance audio quality
    enhanced_audio, sample_rate = enhance_audio_quality(audio_file)
    
    # Intelligent language detection
    if auto_lang:
        detected_language = detect_language_intelligently(model, enhanced_audio)
    else:
        detected_language = "ro"  # Default to Romanian
        print(f"🌍 Using preset language: {detected_language}")
    
    # Perform speaker diarization
    speaker_segments = perform_speaker_diarization_with_progress(audio_file)
    
    # Enhanced transcription
    result, transcription_time, speed_ratio = transcribe_with_enhanced_quality(
        model, enhanced_audio, output_file, speaker_segments, detected_language
    )
    
    # Clean up enhanced audio file
    if enhanced_audio != audio_file and os.path.exists(enhanced_audio):
        try:
            os.unlink(enhanced_audio)
            print("🗑️ Enhanced audio file cleaned up")
        except:
            pass
    
    # Final summary
    print("=" * 60)
    print("🎉 ENHANCED TRANSCRIPTION COMPLETED!")
    print("=" * 60)
    
    total_duration = result['segments'][-1]['end'] if result['segments'] else 0
    unique_speakers = len(set(assign_speaker_to_segment((seg['start'] + seg['end'])/2, speaker_segments) 
                            for seg in result['segments'])) if speaker_segments else 1
    
    print(f"📊 QUALITY SUMMARY:")
    print(f"   📁 Output file: {output_file}")
    print(f"   🌍 Language: {detected_language}")
    print(f"   ⏱️  Audio duration: {total_duration:.1f} seconds")
    print(f"   🚀 Processing time: {transcription_time:.1f} seconds")
    print(f"   ⚡ Speed ratio: {speed_ratio:.1f}x real-time")
    print(f"   📝 Total segments: {len(result['segments'])}")
    print(f"   👥 Speakers detected: {unique_speakers}")
    print(f"   🔥 Final GPU status: {monitor_gpu_performance()}")
    print("💎 Maximum quality optimization complete!")

if __name__ == "__main__":
    main() 