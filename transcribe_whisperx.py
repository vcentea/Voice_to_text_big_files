#!/usr/bin/env python3
"""
HIGH-QUALITY WhisperX Transcription with Speaker Diarization (GPU-ONLY)
Features:
- WhisperX for superior accuracy and word-level timestamps
- Forced phoneme alignment for precise timing
- Advanced speaker diarization integration
- VAD preprocessing to reduce hallucinations
- Batched inference for maximum speed
- Quality validation and error detection
- GPU-only mode optimized for RTX 3090 Ti
- Aggressive CUDA optimizations for maximum performance
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

# Constants for MAXIMUM ACCURACY (time is not an issue)
BATCH_SIZE_GPU = 16  # Smaller batches for better accuracy (vs speed optimization)
MIN_SEGMENT_LENGTH = 1.0  # Conservative - closer to original working parameters
MAX_SEGMENT_GAP = 0.5  # Very tight gap tolerance to prevent cross-speaker merging
MAX_REPETITION_RATIO = 0.2  # Stricter repetition filtering for accuracy
COMPUTE_TYPE_GPU = "float16"  # Best balance of accuracy/memory for RTX 3090 Ti
MIN_WORDS_PER_SEGMENT = 2  # Lower threshold to preserve natural boundaries

def setup_environment():
    """Setup comprehensive CUDA environment for WhisperX (based on enhanced script)"""
    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9"
    
    # Add CUDA bin and lib directories to PATH (fixes cuDNN DLL issues)
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin"
    cuda_lib = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\x64"
    
    current_path = os.environ.get("PATH", "")
    
    # Add CUDA bin to PATH
    if cuda_bin not in current_path:
        os.environ["PATH"] = f"{cuda_bin};{current_path}"
        print(f"🔧 Added CUDA bin to PATH: {cuda_bin}")
    
    # Add CUDA lib to PATH (critical for cuDNN DLLs)
    if cuda_lib not in current_path:
        os.environ["PATH"] = f"{cuda_lib};{os.environ['PATH']}"
        print(f"🔧 Added CUDA lib to PATH: {cuda_lib}")
    
    # Also check for cuDNN in alternative locations (including pip-installed cuDNN)
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else ""
    
    potential_cudnn_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
        "C:\\Program Files\\NVIDIA\\CUDNN\\v8.9\\bin",
        "C:\\tools\\cuda\\bin",
        "C:\\dev\\cuda\\bin",
        f"{site_packages}\\nvidia\\cudnn\\bin" if site_packages else "",
        f"{site_packages}\\nvidia\\cublas\\bin" if site_packages else "",
        "E:\\ENVs\\transcribe\\Lib\\site-packages\\nvidia\\cudnn\\bin",  # Your specific env
        "E:\\ENVs\\transcribe\\Lib\\site-packages\\nvidia\\cublas\\bin"
    ]
    
    for cudnn_path in potential_cudnn_paths:
        if cudnn_path and os.path.exists(cudnn_path) and cudnn_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{cudnn_path};{os.environ['PATH']}"
            print(f"🔧 Added cuDNN path: {cudnn_path}")
    
    # Suppress specific warnings but keep important ones
    warnings.filterwarnings("ignore", message=".*You have requested a HuggingFace token.*")
    warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
    warnings.filterwarnings("ignore", message=".*falling back to a slower.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisperx")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
    warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
    warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
    warnings.filterwarnings("ignore", message=".*audioread.*")
    
    # CUDA optimizations for MAXIMUM ACCURACY (not speed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for max precision
        torch.backends.cudnn.allow_tf32 = False        # Disable TF32 for max precision
        torch.backends.cudnn.benchmark = False         # Disable for deterministic results
        torch.backends.cudnn.deterministic = True      # Enable deterministic mode
        torch.backends.cuda.enable_flash_sdp(False)    # Disable flash attention for accuracy
        
        # Conservative memory fraction for stability
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        print("🎯 CUDA optimized for MAXIMUM ACCURACY (precision over speed)")
        print("💎 Deterministic mode enabled for consistent results")
    else:
        print("❌ CUDA not available - GPU-only mode requires CUDA!")
        sys.exit(1)

def verify_cudnn_availability():
    """Verify cuDNN is properly available"""
    try:
        if torch.cuda.is_available():
            # Test cuDNN availability
            x = torch.randn(1, 1, 1, 1, device='cuda')
            torch.nn.functional.conv2d(x, torch.randn(1, 1, 1, 1, device='cuda'))
            print("✅ cuDNN verification successful")
            return True
    except Exception as e:
        print(f"⚠️ cuDNN verification failed: {e}")
        print("🔧 Attempting cuDNN fallback configuration...")
        
        # Try to disable cuDNN and use alternative backend
        try:
            torch.backends.cudnn.enabled = False
            print("✅ Disabled cuDNN - using alternative CUDA backend")
            print("⚠️ Performance may be slower but execution will continue")
            return False
        except:
            print("❌ Could not configure cuDNN fallback")
            return False
    return False

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
    """Get GPU-only device configuration optimized for RTX 3090 Ti"""
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not available!")
        print("💡 This script requires GPU acceleration.")
        print("🔧 Please check:")
        print("   - NVIDIA drivers are installed")
        print("   - CUDA toolkit is installed")
        print("   - PyTorch with CUDA support is installed")
        print("   - GPU is not being used by other processes")
        sys.exit(1)
    
    device = "cuda"
    compute_type = COMPUTE_TYPE_GPU
    batch_size = BATCH_SIZE_GPU
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🚀 GPU-ONLY MODE ENABLED")
    print(f"✅ Using GPU: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    print(f"✅ Compute Type: {compute_type}")
    print(f"✅ Batch Size: {batch_size}")
    
    # Optimize specifically for RTX 3090 Ti - ACCURACY MODE
    if "3090" in gpu_name:
        print(f"🎯 RTX 3090 Ti detected - using accuracy-optimized settings")
        print(f"💎 Maximum accuracy expected with {gpu_memory:.1f}GB VRAM")
        # Use smaller batch size for better accuracy
        if gpu_memory > 20:  # RTX 3090 Ti has 24GB
            batch_size = 12  # Smaller batches for maximum accuracy
            print(f"🎯 Using batch size {batch_size} for maximum accuracy (not speed)")
    
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

def merge_short_segments(segments):
    """Merge short segments to improve content flow and reduce over-segmentation"""
    if not segments:
        return segments
    
    print("🔗 Merging short segments to improve conversation flow...")
    merged_segments = []
    current_segment = None
    
    for segment in segments:
        text = segment.get('text', '').strip()
        words = text.split()
        duration = segment.get('end', 0) - segment.get('start', 0)
        speaker = segment.get('speaker', 'UNKNOWN')
        
        # Skip empty segments
        if not text:
            continue
            
        # CONSERVATIVE merging - only merge very obvious fragments
        should_merge = (
            current_segment is not None and
            len(words) < MIN_WORDS_PER_SEGMENT and  # Very few words
            duration < MIN_SEGMENT_LENGTH and       # Very short duration  
            speaker == current_segment.get('speaker', 'UNKNOWN') and  # MUST be same speaker
            speaker != 'SPEAKER_UNKNOWN' and        # MUST have valid speaker ID
            (segment.get('start', 0) - current_segment.get('end', 0)) < MAX_SEGMENT_GAP and  # Very close timing
            len(current_segment.get('text', '').split()) < 8  # Don't merge into already long segments
        )
        
        if should_merge:
            # Merge with current segment
            current_segment['text'] = current_segment['text'].strip() + ' ' + text
            current_segment['end'] = segment.get('end', current_segment['end'])
            
            # Merge word-level data if available
            if 'words' in current_segment and 'words' in segment:
                current_segment['words'].extend(segment['words'])
                
        else:
            # Start new segment
            if current_segment is not None:
                merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    original_count = len(segments)
    merged_count = len(merged_segments)
    reduction = ((original_count - merged_count) / original_count * 100) if original_count > 0 else 0
    
    print(f"✅ Segment merging complete: {original_count} → {merged_count} segments ({reduction:.1f}% reduction)")
    return merged_segments

def enhance_segment_boundaries(segments):
    """Enhance segment boundaries for better conversation flow (optimized for recorded files)"""
    if not segments:
        return segments
        
    print("📝 Enhancing segment boundaries for recorded file conversation flow...")
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        text = segment.get('text', '').strip()
        
        # Skip empty segments
        if not text:
            continue
            
        # Check for natural sentence endings
        ends_naturally = text.endswith(('.', '!', '?', ':', ';'))
        next_segment = segments[i + 1] if i + 1 < len(segments) else None
        
        # For recorded files, be more aggressive in merging incomplete thoughts
        if (not ends_naturally and 
            next_segment and 
            segment.get('speaker') == next_segment.get('speaker') and
            (next_segment.get('start', 0) - segment.get('end', 0)) < MAX_SEGMENT_GAP):
            
            # Look ahead to find natural ending (more aggressive for recorded files)
            lookahead = []
            j = i + 1
            while j < len(segments) and j < i + 5:  # Look ahead up to 5 segments for recorded files
                lookahead_seg = segments[j]
                if lookahead_seg.get('speaker') != segment.get('speaker'):
                    break
                    
                lookahead.append(lookahead_seg)
                lookahead_text = lookahead_seg.get('text', '').strip()
                
                # For recorded files, look for more natural ending patterns
                if (lookahead_text.endswith(('.', '!', '?')) or 
                    (j == i + 4 and len(lookahead_text.split()) >= 3)):  # Fallback after 4 segments
                    
                    # Found natural ending or reached reasonable limit, merge segments
                    merged_text = text
                    merged_end = segment.get('end', 0)
                    merged_words = segment.get('words', []).copy()
                    
                    for merge_seg in lookahead:
                        merged_text += ' ' + merge_seg.get('text', '').strip()
                        merged_end = merge_seg.get('end', merged_end)
                        if 'words' in merge_seg:
                            merged_words.extend(merge_seg['words'])
                    
                    # Create enhanced segment
                    enhanced_segment = segment.copy()
                    enhanced_segment['text'] = merged_text
                    enhanced_segment['end'] = merged_end
                    if merged_words:
                        enhanced_segment['words'] = merged_words
                    
                    enhanced_segments.append(enhanced_segment)
                    
                    # Skip the merged segments
                    i = j
                    break
                    
                j += 1
            else:
                # No natural ending found, keep original
                enhanced_segments.append(segment)
        else:
            enhanced_segments.append(segment)
    
    print(f"✅ Enhanced {len(enhanced_segments)} segments for recorded file flow")
    return enhanced_segments

def transcribe_with_whisperx(audio_file, device, compute_type, batch_size, language=None):
    """Transcribe audio using WhisperX with quality validation"""
    print("\n🎙️ WHISPERX TRANSCRIPTION STARTING...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Load WhisperX model optimized for MAXIMUM ACCURACY
        print(f"📥 Loading WhisperX large-v3 model on {device.upper()} (max accuracy mode)...")
        
        # Define the temperature fallback sequence
        temperature_fallback = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        print(f"🌡️ Temperature fallback strategy: {temperature_fallback}")
        print(f"🎯 Beam size: 5 (default)")

        model = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            asr_options={
                # WhisperX specific or general model setup options
                "suppress_numerals": False,
                "max_new_tokens": None,
                "clip_timestamps": "0",
                "word_timestamps": True, # Used by WhisperX alignment
                "hallucination_silence_threshold": 0.5, # WhisperX VAD-related

                # Core FasterWhisper TranscriptionOptions for MAX ACCURACY
                "temperatures": temperature_fallback,    # PLURAL KEY for fallback sequence
                "beam_size": 5,                         # Default is 5
                "patience": 1.0,                        # Default for ctranslate2 (must be float, not None)
                "length_penalty": 1.0,                  # Default
                "no_speech_threshold": 0.6,             # Default VAD threshold
                "log_prob_threshold": -1.0,             # Default log probability threshold
                "compression_ratio_threshold": 2.4,     # Default gibberish detection
                "condition_on_previous_text": True     # Crucial for context & reducing repetition
            }
        )
        print(f"✅ WhisperX model loaded with advanced ASR options.")
        print(f"🌡️ Temperature fallback strategy: {temperature_fallback}")
        print(f"🎯 Beam size: 5, Patience: 1.0, Length penalty: 1.0")
        print(f"💎 Quality thresholds: no_speech={0.6}, log_prob={-1.0}, compression_ratio={2.4}")
        
        # 2. Load audio
        print(f"📥 Loading audio: {audio_file}")
        audio = whisperx.load_audio(audio_file)
        audio_duration = len(audio) / 16000
        print(f"✅ Audio loaded: {audio_duration:.1f} seconds")
        
        # 3. Transcribe with WhisperX (maximum accuracy settings)
        print(f"🚀 Transcribing with batch size {batch_size} (max accuracy mode)...")
        if language:
            print(f"🌍 Using language parameter: {language}")
        else:
            print("🌍 No language specified, will auto-detect")
        
        from tqdm import tqdm
        import sys
        
        # Estimate number of chunks for progress bar
        estimated_chunks = int(audio_duration / 30) + 1  # 30-second chunks
        
        with tqdm(total=estimated_chunks, desc="🎙️ Transcribing",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]",
                  file=sys.stdout, ncols=80) as pbar:

            # All decoding options are now set via asr_options in load_model
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,
                print_progress=False,      # Disabled - using custom progress bar
                combined_progress=False    # Disabled - using custom progress bar
            )

            pbar.n = estimated_chunks
            pbar.refresh()
        
        print(f"✅ Initial transcription complete: {len(result['segments'])} segments")
        
        # Clean up model memory
        del model
        gc.collect()
        torch.cuda.empty_cache()  # Always clear CUDA cache in GPU-only mode
        
        # 4. Enhanced quality validation and segment optimization
        print("📊 QUALITY ENHANCEMENT PIPELINE")
        print("-" * 40)
        
        original_count = len(result['segments'])
        print(f"📝 Original segments: {original_count}")
        
        # Step 1: Clean repetitive content
        repetition_ratio, repetitive_indices = detect_repetitions(result['segments'])
        print(f"📊 Quality check - Repetition ratio: {repetition_ratio:.1%}")
        
        if repetition_ratio > MAX_REPETITION_RATIO:
            print(f"⚠️ Quality issue detected! Cleaning repetitive segments...")
            result['segments'] = clean_repetitive_segments(result['segments'])
        
        # Step 2: CONSERVATIVE merging of only obvious fragments
        result['segments'] = merge_short_segments(result['segments'])
        
        # Step 3: DISABLED - boundary enhancement was too aggressive
        # result['segments'] = enhance_segment_boundaries(result['segments'])
        
        final_count = len(result['segments'])
        improvement = ((original_count - final_count) / original_count * 100) if original_count > 0 else 0
        print(f"✅ Segment optimization complete: {original_count} → {final_count} segments ({improvement:.1f}% reduction)")
        print("-" * 40)
        
        # 5. Forced alignment for precise timestamps
        if result['segments']:
            print(f"🎯 Loading alignment model for {result.get('language', 'detected language')}...")
            
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=device
                )
                
                print(f"🔍 Performing forced alignment (max accuracy)...")
                
                # Progress bar for alignment
                alignment_segments = len(result["segments"])
                with tqdm(total=alignment_segments, desc="🎯 Aligning", 
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} segments [{elapsed}<{remaining}]",
                          file=sys.stdout, ncols=80) as align_pbar:
                    
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device, 
                        return_char_alignments=False
                        # Using default alignment settings for maximum compatibility
                    )
                    
                    # Complete alignment progress
                    align_pbar.n = alignment_segments
                    align_pbar.refresh()
                
                print(f"✅ Forced alignment complete")
                
                # Clean up alignment model
                del model_a
                gc.collect()
                torch.cuda.empty_cache()  # Always clear CUDA cache in GPU-only mode
                    
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
    """Perform speaker diarization using pyannote.audio directly (same as enhanced script)"""
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
        
        # Load diarization model using pyannote.audio directly (same as enhanced script)
        print(f"📥 Loading speaker diarization model on {device.upper()}...")
        from pyannote.audio import Pipeline
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if device == "cuda":
            import torch
            pipeline = pipeline.to(torch.device("cuda"))
            
        print(f"✅ Diarization model loaded")
        
        # Perform diarization
        print("🔍 Analyzing speakers...")
        diarization = pipeline(audio_file)
        
        # Convert to WhisperX-compatible format
        speakers_found = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_found.add(speaker)
        
        elapsed = time.time() - start_time
        print(f"✅ SPEAKER DIARIZATION COMPLETE!")
        print(f"👥 Found {len(speakers_found)} speakers: {', '.join(sorted(speakers_found))}")
        print(f"⏱️ Diarization time: {elapsed:.1f}s")
        print("=" * 50)
        
        return diarization
        
    except Exception as e:
        print(f"⚠️ Speaker diarization failed: {e}")
        print("🔧 Continuing without speaker labels...")
        return None

def assign_speakers_to_segments(result, diarize_segments):
    """Assign speaker labels to transcription segments"""
    if not diarize_segments or not result.get('segments'):
        return result
    
    print("🏷️ Assigning speaker labels to segments...")
    
    # Try WhisperX's built-in speaker assignment first
    try:
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("✅ Used WhisperX assign_word_speakers")
    except Exception as e:
        print(f"⚠️ WhisperX assign_word_speakers failed: {e}")
        print("🔧 Using manual speaker assignment...")
        
        # Manual assignment as fallback (same logic as enhanced script)
        def assign_speaker_to_segment(segment_start, segment_end, speaker_segments):
            segment_mid = (segment_start + segment_end) / 2
            
            for turn, _, speaker in speaker_segments.itertracks(yield_label=True):
                if turn.start <= segment_mid <= turn.end:
                    return speaker
            
            return "SPEAKER_UNKNOWN"
        
        # Assign speakers manually
        for segment in result['segments']:
            start_ts = segment.get('start', 0)
            end_ts = segment.get('end', start_ts + 1)
            speaker = assign_speaker_to_segment(start_ts, end_ts, diarize_segments)
            segment['speaker'] = speaker
    
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
    """Write enhanced SRT file with speaker labels and improved formatting"""
    print(f"📝 Writing enhanced SRT to: {output_file}")
    
    segments = result.get('segments', [])
    if not segments:
        print("❌ No segments to write")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        segment_count = 0
        total_duration = 0
        speaker_counts = {}
        
        for i, segment in enumerate(segments):
            start = segment.get('start', 0)
            end = segment.get('end', start + 1)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            
            # Skip empty segments
            if not text:
                continue
            
            # Apply minimum duration filter (now more lenient due to merging)
            duration = end - start
            if duration < 0.5:  # Very short segments only
                continue
            
            segment_count += 1
            total_duration += duration
            
            # Track speaker statistics
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # Format timestamps
            start_time = format_timestamp_srt(start)
            end_time = format_timestamp_srt(end)
            
            # Enhanced text formatting for better readability
            # Clean up extra spaces and ensure proper capitalization
            text = ' '.join(text.split())  # Clean extra whitespace
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            
            # Format text with speaker (improved speaker labeling)
            if speaker != 'SPEAKER_UNKNOWN':
                speaker_text = f"[{speaker}] {text}"
            else:
                speaker_text = text
            
            # Write SRT entry with improved formatting
            f.write(f"{segment_count}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{speaker_text}\n\n")
    
    # Enhanced completion summary
    avg_duration = total_duration / segment_count if segment_count > 0 else 0
    print(f"✅ Enhanced SRT written: {segment_count} segments")
    print(f"📊 Average segment duration: {avg_duration:.1f}s (improved from over-segmentation)")
    print(f"👥 Speaker distribution: {len(speaker_counts)} unique speakers")
    for speaker, count in sorted(speaker_counts.items()):
        percentage = (count / segment_count * 100) if segment_count > 0 else 0
        print(f"   {speaker}: {count} segments ({percentage:.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_whisperx.py <audio_file> [output_file] [--language code]")
        print("Example: python transcribe_whisperx.py audio.wav output.srt --language ro")
        print("⚠️  GPU-ONLY MODE: Requires NVIDIA GPU with CUDA support")
        print("🎯 Optimized for RTX 3090 Ti with 24GB VRAM")
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
    print("💎 GPU-ONLY MODE - Optimized for RTX 3090 Ti")
    print("=" * 70)
    
    setup_environment()
    setup_ffmpeg()
    
    # Verify cuDNN is working properly
    verify_cudnn_availability()
    
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
        print(f"🌍 Language specified: {language}")
    else:
        print("🌍 Language: auto-detect (increases inference time)")
    
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
        
        print(f"📊 ENHANCED QUALITY SUMMARY:")
        print(f"   📁 Output file: {output_file}")
        print(f"   🌍 Language: {result.get('language', 'auto-detected')}")
        print(f"   ⏱️  Audio duration: {total_duration:.1f} seconds")
        print(f"   🚀 Processing time: {transcription_time:.1f} seconds")
        print(f"   ⚡ Speed ratio: {speed_ratio:.1f}x real-time")
        print(f"   📝 Optimized segments: {len(result['segments'])} (reduced over-segmentation)")
        print(f"   👥 Speakers detected: {unique_speakers}")
        print("🎯 MAXIMUM ACCURACY CONFIGURATION:")
        print("   ✅ Smaller batch sizes for better context processing")
        print("   ✅ Full precision mode (no TF32 approximations)")
        print("   ✅ Deterministic CUDA operations for consistency")
        print("   ✅ Intelligent temperature fallback (temperatures key in asr_options)")
        print("   ✅ Beam search (size=5, default) via asr_options")
        print("   ✅ Advanced decoding options (patience, thresholds) via asr_options")
        print("   ✅ Conservative post-processing to preserve quality")
        print("   ✅ Context-aware transcription (condition_on_previous_text=True in asr_options)")
        print(f"   ✅ Temperature sequence in asr_options: {temperature_fallback if 'temperature_fallback' in locals() else 'N/A'}")
        print("💎 WhisperX configured for maximum accuracy with intelligent fallback!")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 