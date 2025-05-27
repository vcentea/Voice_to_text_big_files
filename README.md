# Whisper Transcription Project

## Project Overview

This project provides advanced speech-to-text transcription capabilities using OpenAI's Whisper models with two main implementations:

1. **transcribe_enhanced.py** - Enhanced transcription with audio preprocessing and speaker diarization
2. **whisper_server.py** - FastAPI-based web server for transcription services

Both scripts support GPU acceleration with CUDA and include robust fallback mechanisms.

---

## Core Scripts Documentation

### 1. transcribe_enhanced.py

**Purpose**: Advanced transcription with maximum quality optimization, audio preprocessing, and speaker diarization.

#### Key Features:
- **Audio Enhancement**: Noise reduction, normalization, format conversion
- **Speaker Diarization**: Automatic speaker identification using pyannote.audio
- **Smart Language Detection**: Intelligent language detection with confidence scoring
- **Quality Optimization**: Advanced Whisper parameters (temperature, beam search)
- **GPU Optimization**: CUDA acceleration with automatic CPU fallback
- **Real-time Progress**: Live progress monitoring with performance metrics

#### Architecture:
```
Audio Input → Audio Enhancement → Language Detection → Whisper Transcription → Speaker Diarization → Output SRT
```

#### Dependencies:
- **Core Libraries**:
  - `whisper` - OpenAI Whisper for ASR
  - `torch` - PyTorch for GPU/CPU operations
  - `numpy` - Numerical operations
  - `librosa` - Audio processing and loading
  - `noisereduce` - Advanced noise reduction
  - `soundfile` - Audio file I/O

- **Audio Processing**:
  - `subprocess` - FFmpeg integration for format conversion
  - `tempfile` - Temporary file management

- **Speaker Diarization**:
  - `pyannote.audio` - Speaker diarization pipeline

- **Environment & Utilities**:
  - `python-dotenv` - Environment variable loading
  - `pathlib` - Path operations
  - Standard library: `os`, `sys`, `time`, `warnings`, `threading`

#### Configuration:
- **Environment Variables**: `HF_TOKEN` (required for speaker diarization)
- **CUDA Setup**: Automatic detection with optimized settings
- **FFmpeg**: Required for audio format conversion

#### Usage:
```bash
python transcribe_enhanced.py <audio_file> [output_file] [--auto-lang]
```

#### Performance Optimizations:
- **CUDA Optimizations**: TF32, cuDNN benchmark mode
- **Memory Management**: Automatic GPU cache clearing
- **Chunked Processing**: Efficient audio processing in segments
- **Real-time Monitoring**: GPU utilization and speed metrics

---

### 2. whisper_server.py

**Purpose**: FastAPI-based REST API server for transcription services with faster-whisper backend.

#### Key Features:
- **REST API**: Complete web service with endpoints
- **Model Management**: Dynamic model loading and switching
- **High Performance**: Uses faster-whisper (CTranslate2 backend)
- **GPU Acceleration**: Optimized for NVIDIA GPUs
- **Multiple Formats**: Supports various audio formats
- **Health Monitoring**: System status and performance metrics

#### Architecture:
```
HTTP Request → Audio Upload → Format Processing → faster-whisper → JSON Response
```

#### API Endpoints:

1. **POST /transcribe**
   - Upload audio file for transcription
   - Parameters: `file` (audio), `model` (optional)
   - Returns: JSON with text, segments, timing, and metadata

2. **GET /models**
   - List available models and current configuration
   - Returns: Available models, GPU status, compute type

3. **POST /change_model**
   - Switch transcription model
   - Parameters: `model_name`
   - Returns: Success/error status

4. **GET /health**
   - System health check
   - Returns: Status, model info, GPU metrics

#### Dependencies:
- **Web Framework**:
  - `fastapi` - Modern web framework
  - `uvicorn` - ASGI server
  - `pydantic` - Data validation

- **Transcription Engine**:
  - `faster-whisper` - CTranslate2-based Whisper implementation
  - `torch` - PyTorch for GPU operations

- **Audio Processing**:
  - `soundfile` - Audio file loading
  - `scipy` - Signal processing (resampling)
  - `numpy` - Array operations

- **Utilities**:
  - Standard library: `os`, `sys`, `time`, `json`, `warnings`
  - `typing` - Type annotations
  - `io.BytesIO` - Binary data handling

#### Configuration:
- **Server Settings**: 
  - Host: `0.0.0.0` (all interfaces)
  - Port: `5555`
  - Default model: `large-v3`

- **GPU Settings**:
  - Automatic CUDA detection
  - Compute type: `float16` (GPU) / `int8` (CPU)
  - Memory optimization

#### Performance Features:
- **Model Caching**: Pre-loaded models for faster response
- **GPU Memory Management**: Automatic cache clearing
- **Batch Processing**: Optimized for multiple requests
- **CTranslate2 Backend**: Up to 4x faster than standard Whisper

#### Usage:
```bash
python whisper_server.py [model_name]
```

Access API at: `http://localhost:5555`

---

## System Requirements

### Hardware:
- **GPU (Recommended)**: NVIDIA RTX 3090 Ti or equivalent
- **VRAM**: 12GB+ for large-v3 model
- **RAM**: 16GB+ system memory
- **CPU**: Multi-core processor (fallback mode)

### Software:
- **Python**: 3.8+
- **CUDA**: 12.1+ (for GPU acceleration)
- **cuDNN**: 8.9+ (for GPU operations)
- **FFmpeg**: Required for audio processing

### Platform:
- **Development**: Windows 11
- **Production**: Ubuntu (recommended)
- **Environment**: Python virtual environment recommended

---

## Installation Guide

### Prerequisites

#### System Requirements:
- **Python 3.8+** 
- **Git** for version control
- **FFmpeg** for audio processing
- **CUDA 12.1+** and **cuDNN 8.9.7.29** (required for GPU acceleration)

#### Hardware Recommendations:
- **GPU**: NVIDIA RTX 3090 Ti or equivalent (12GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models

### Detailed Installation Guide

This guide includes solutions to common issues we encountered during setup.

#### 1. Clone the Repository
```bash
git clone https://github.com/vcentea/Voice_to_text_big_files.git
cd Voice_to_text_big_files
```

#### 2. Create Virtual Environment (CRITICAL STEP)
```bash
# Windows - Create in a dedicated location for better organization
python -m venv E:\ENVs\transcribe
E:\ENVs\transcribe\Scripts\activate

# Alternative: Create in project folder
python -m venv transcribe
transcribe\Scripts\activate

# Linux/macOS
python3 -m venv transcribe
source transcribe/bin/activate
```

**⚠️ Important Virtual Environment Notes:**
- Always activate your virtual environment before installing packages
- Use a consistent virtual environment for all packages
- If you see import errors, verify you're in the correct environment

#### 3. Install Core Dependencies (SPECIFIC ORDER MATTERS)

**Step 3a: Install PyTorch with CUDA Support FIRST**
```bash
# For CUDA 12.1 (REQUIRED for RTX 3090 Ti)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Step 3b: Install cuDNN 8 (CRITICAL VERSION COMPATIBILITY)**
```bash
# Install EXACTLY this version - cuDNN 9 will NOT work with WhisperX
pip install nvidia-cudnn-cu12==8.9.7.29
pip install nvidia-cublas-cu12>=12.1.0

# Verify installation
python -c "import site; print([p for p in site.getsitepackages()])"
```

**Step 3c: Install python-dotenv (Required for .env file loading)**
```bash
pip install python-dotenv>=1.0.0
```

**Step 3d: Install WhisperX and Dependencies**
```bash
# Install all remaining packages
pip install -r requirements.txt
```

#### 4. Setup Environment Variables (.env file)

**Critical for Speaker Diarization to Work**

Create a `.env` file in the project root directory:

**Method 1: Copy from example**
```bash
# Copy the example file
copy .env.example .env
# Then edit .env and replace 'your_huggingface_token_here' with your actual token
```

**Method 2: Create manually**
```bash
# Windows
echo HF_TOKEN=your_huggingface_token_here > .env

# Linux/macOS  
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

**Get your HuggingFace token (REQUIRED STEPS):**
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **read** permissions
3. **CRITICAL**: Accept the license for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **CRITICAL**: Accept the license for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Verify your .env file:**
Your `.env` file should contain:
```
HF_TOKEN=hf_your_actual_token_here
```
⚠️ **Note**: Keep your token private and never commit the `.env` file to version control!

**Test .env loading:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token loaded:', 'Yes' if os.getenv('HF_TOKEN') else 'No')"
```

#### 5. Install FFmpeg (Required for Audio Processing)

**Windows:**
```bash
# Method 1: Using winget (recommended)
winget install --id=Gyan.FFmpeg -e

# Method 2: Manual installation
# 1. Download from https://ffmpeg.org/download.html
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to your PATH environment variable
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Verify FFmpeg installation:**
```bash
ffmpeg -version
```

#### 6. Critical cuDNN Setup (GPU-Only Mode)

**Why cuDNN 8 is Required:**
- WhisperX and PyTorch are compiled for cuDNN 8
- cuDNN 9 has different DLL names and will cause "cudnn_ops_infer64_8.dll not found" errors
- Our scripts automatically add cuDNN paths to system PATH

**Verify cuDNN Installation:**
```bash
# Check if cuDNN 8 DLLs exist (Windows)
dir "E:\ENVs\transcribe\Lib\site-packages\nvidia\cudnn\bin\cudnn_ops_infer64_8.dll"

# Or check your specific virtual environment path
python -c "import site; print(site.getsitepackages()[0] + '\\nvidia\\cudnn\\bin')"
```

#### 7. Verification Steps

**Test 1: GPU and CUDA**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected Output:**
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3090 Ti
```

**Test 2: Environment Variables**
```bash
python transcribe_whisperx.py
```

**Expected Output:**
```
✅ Environment variables loaded from .env file
✅ HF_TOKEN found in .env: hf_tsxOCcW...
✅ cuDNN verification successful
🚀 GPU-ONLY MODE ENABLED
✅ Using GPU: NVIDIA GeForce RTX 3090 Ti
✅ GPU Memory: 24.0 GB
```

**Test 3: Full Transcription Test**
```bash
# Test with a short audio file
python transcribe_whisperx.py test_audio.wav output.srt --language en
```

### Common Issues and Solutions

#### Issue 1: "Could not locate cudnn_ops_infer64_8.dll"

**Cause:** Wrong cuDNN version installed (version 9 instead of 8)

**Solution:**
```bash
# Uninstall cuDNN 9
pip uninstall nvidia-cudnn-cu12

# Install cuDNN 8 specifically
pip install nvidia-cudnn-cu12==8.9.7.29

# Verify the correct DLLs exist
python -c "import os, site; print(os.path.exists(site.getsitepackages()[0] + '\\nvidia\\cudnn\\bin\\cudnn_ops_infer64_8.dll'))"
```

#### Issue 2: "python-dotenv not installed"

**Cause:** Missing package for .env file loading

**Solution:**
```bash
pip install python-dotenv>=1.0.0
```

#### Issue 3: "HF_TOKEN not found"

**Cause:** Missing or incorrect .env file setup

**Solution:**
1. Create `.env` file in project root
2. Add `HF_TOKEN=your_token_here` (no quotes)
3. Accept HuggingFace model licenses
4. Test: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('HF_TOKEN')[:10] if os.getenv('HF_TOKEN') else 'NOT_FOUND')"`

#### Issue 4: "GPU not available" or "CUDA not available"

**Cause:** PyTorch installed without CUDA support

**Solution:**
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchaudio

# Install CUDA-enabled PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 5: "Model loading fails" or "Import errors"

**Cause:** Virtual environment issues or missing dependencies

**Solution:**
```bash
# Ensure you're in the correct virtual environment
which python  # Should point to your virtual environment

# Reinstall all requirements
pip install --force-reinstall -r requirements.txt
```

#### Issue 6: "FFmpeg not found"

**Cause:** FFmpeg not installed or not in PATH

**Solution:**
```bash
# Windows: Download and add to PATH, or use winget
winget install --id=Gyan.FFmpeg -e

# Test installation
ffmpeg -version
```

### Performance Optimization for RTX 3090 Ti

**Optimal Settings (Already configured in scripts):**
- **Batch Size**: 32 (auto-detected for 24GB VRAM)
- **Compute Type**: float16 (maximum speed)
- **Memory Usage**: 95% of GPU memory
- **cuDNN Optimizations**: Enabled with TF32
- **Flash Attention**: Enabled when available

**Expected Performance:**
- **Speed**: 80-100x real-time transcription
- **Quality**: Superior accuracy with forced alignment
- **Memory**: ~8-12GB VRAM usage for large-v3 model
- **Processing**: ~1 hour audio processed in 30-60 seconds

### Complete Installation Workflow (Recommended)

Follow this exact sequence to avoid the issues we encountered:

```bash
# 1. Clone repository
git clone https://github.com/vcentea/Voice_to_text_big_files.git
cd Voice_to_text_big_files

# 2. Create virtual environment
python -m venv E:\ENVs\transcribe  # or your preferred location
E:\ENVs\transcribe\Scripts\activate

# 3. Install PyTorch with CUDA FIRST
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install cuDNN 8 (EXACT VERSION)
pip install nvidia-cudnn-cu12==8.9.7.29
pip install nvidia-cublas-cu12>=12.1.0

# 5. Install python-dotenv
pip install python-dotenv>=1.0.0

# 6. Install all other requirements
pip install -r requirements.txt

# 7. Create .env file
echo HF_TOKEN=your_huggingface_token_here > .env

# 8. Install FFmpeg (Windows)
winget install --id=Gyan.FFmpeg -e

# 9. Test installation
python transcribe_whisperx.py
```

### Installation Verification Checklist

Run these commands to verify everything is working:

**✅ Step 1: Virtual Environment**
```bash
python -c "import sys; print('Python path:', sys.executable)"
# Should point to your virtual environment
```

**✅ Step 2: PyTorch and CUDA**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**✅ Step 3: cuDNN 8 Verification**
```bash
python -c "import os, site; cudnn_path = site.getsitepackages()[0] + '\\nvidia\\cudnn\\bin\\cudnn_ops_infer64_8.dll'; print('cuDNN 8 DLL exists:', os.path.exists(cudnn_path))"
```

**✅ Step 4: Environment Variables**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF_TOKEN loaded:', 'Yes' if os.getenv('HF_TOKEN') else 'No')"
```

**✅ Step 5: WhisperX Import Test**
```bash
python -c "import whisperx; print('WhisperX import: OK')"
```

**✅ Step 6: FFmpeg Test**
```bash
ffmpeg -version
```

**✅ Step 7: Full System Test**
```bash
python transcribe_whisperx.py
# Should show GPU detection and no errors
```

### Troubleshooting Guide

#### Problem: "Could not locate cudnn_ops_infer64_8.dll"

**Diagnosis:**
```bash
# Check current cuDNN version
pip list | findstr cudnn
# If it shows version 9.x, that's the problem
```

**Solution:**
```bash
pip uninstall nvidia-cudnn-cu12
pip install nvidia-cudnn-cu12==8.9.7.29
```

#### Problem: "python-dotenv not installed"

**Solution:**
```bash
pip install python-dotenv>=1.0.0
```

#### Problem: "HF_TOKEN not found"

**Diagnosis:**
```bash
python -c "import os; print('.env file exists:', os.path.exists('.env'))"
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token value:', os.getenv('HF_TOKEN', 'NOT_FOUND')[:15])"
```

**Solution:**
1. Create `.env` file in project root
2. Add `HF_TOKEN=hf_your_token_here` (no quotes)
3. Accept licenses at: https://huggingface.co/pyannote/speaker-diarization-3.1

#### Problem: "CUDA not available"

**Diagnosis:**
```bash
python -c "import torch; print('PyTorch CUDA built:', torch.version.cuda)"
python -c "import torch; print('GPU detected:', torch.cuda.is_available())"
```

**Solution:**
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Problem: "FFmpeg not found"

**Solution (Windows):**
```bash
winget install --id=Gyan.FFmpeg -e
# Or manually download and add to PATH
```

#### Problem: Virtual Environment Issues

**Solution:**
```bash
# Deactivate current environment
deactivate

# Remove old environment
rmdir /s "path_to_your_venv"

# Start fresh
python -m venv E:\ENVs\transcribe_fresh
E:\ENVs\transcribe_fresh\Scripts\activate

# Follow installation workflow again
```

### Post-Installation Notes

**For RTX 3090 Ti Users:**
- The script automatically detects your GPU and optimizes settings
- Expect 80-100x real-time transcription speed
- ~8-12GB VRAM usage for large-v3 model
- Batch size automatically set to 32 for maximum performance

**Memory Management:**
- The script uses 95% of GPU memory for maximum performance
- If you encounter OOM errors, the script will automatically fallback
- Monitor GPU usage with `nvidia-smi` during processing

**Quality Settings:**
- WhisperX provides superior accuracy vs standard Whisper
- Forced alignment gives precise word-level timestamps
- Speaker diarization requires HF_TOKEN and license acceptance

---

## File Structure

### Essential Files:
- `transcribe_enhanced.py` - Enhanced transcription script
- `whisper_server.py` - FastAPI web server
- `.env` - Environment variables (HF_TOKEN)
- `models/` - Downloaded Whisper models
- `README.md` - This documentation

### Generated Files:
- `*.srt` - Subtitle output files
- `enhanced_*.srt` - Enhanced transcription outputs
- Model cache files in `models/` directory

---

## Environment Variables

### Required:
- `HF_TOKEN` - HuggingFace token for speaker diarization models

### Optional:
- `CUDA_PATH` - CUDA installation path
- `CUDA_HOME` - CUDA home directory
- Custom model paths and cache directories

---

## Error Handling

### GPU Issues:
- Automatic CPU fallback for CUDA errors
- cuDNN compatibility warnings
- Memory management and cache clearing

### Audio Issues:
- Format conversion with FFmpeg
- Codec compatibility handling
- Sample rate normalization

### Model Issues:
- Automatic model downloading
- Model caching and validation
- Version compatibility checks

---

## Performance Benchmarks

### Typical Performance (RTX 3090 Ti):
- **Speed**: 10-15x real-time transcription
- **Quality**: Large-v3 model with enhanced preprocessing
- **Memory**: ~8GB VRAM usage for large-v3
- **Accuracy**: 95%+ for clear audio with speaker diarization

### Optimization Tips:
1. Use GPU acceleration when available
2. Pre-process audio for better quality
3. Cache models for faster startup
4. Use appropriate batch sizes for your GPU
5. Monitor GPU memory usage

---

## Troubleshooting

### Common Issues:
1. **cuDNN errors**: Use CPU fallback or update drivers
2. **FFmpeg not found**: Install FFmpeg and add to PATH
3. **HF_TOKEN missing**: Add token to .env file
4. **Model download fails**: Check internet connection and HF token
5. **GPU memory errors**: Reduce batch size or use CPU

### Debug Commands:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check models
python -c "import whisper; print(whisper.available_models())"

# Test transcription
python transcribe_enhanced.py sample.wav --cpu
``` 