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
- **CUDA 12.1+** and **cuDNN 8.9+** (optional, for GPU acceleration)

#### Hardware Recommendations:
- **GPU**: NVIDIA RTX 3090 Ti or equivalent (12GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models

### Quick Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/vcentea/Voice_to_text_big_files.git
cd Voice_to_text_big_files
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv transcribe
transcribe\Scripts\activate

# Linux/macOS
python3 -m venv transcribe
source transcribe/bin/activate
```

#### 3. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Setup Environment Variables
Create a `.env` file in the project root:
```bash
# Windows
echo HF_TOKEN=your_huggingface_token_here > .env

# Linux/macOS  
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

**Get your HuggingFace token:**
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Accept the license for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

#### 5. Install FFmpeg

**Windows:**
```bash
# Using winget
winget install --id=Gyan.FFmpeg -e

# Or download from https://ffmpeg.org/download.html
# Add FFmpeg to your PATH environment variable
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

### Detailed Installation Steps

#### Python Virtual Environment Setup

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system packages
- Makes deployment reproducible

**Creating and managing the environment:**
```bash
# Create virtual environment
python -m venv transcribe

# Activate (Windows)
transcribe\Scripts\activate

# Activate (Linux/macOS)
source transcribe/bin/activate

# Verify activation (should show transcribe in prompt)
which python  # Should point to transcribe/bin/python

# Deactivate when done
deactivate
```

#### GPU Setup (Optional but Recommended)

**CUDA Installation:**
1. Download [CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads)
2. Install following NVIDIA's instructions
3. Verify installation: `nvcc --version`

**cuDNN Setup:**
1. Download [cuDNN 8.9+](https://developer.nvidia.com/cudnn)
2. Extract and copy files to CUDA directory
3. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**PyTorch GPU Installation:**
```bash
# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Installation Dependencies

#### Core Python Packages:
```
torch>=2.0.0
torchaudio>=2.0.0
whisper-openai
faster-whisper
fastapi
uvicorn
pydantic
numpy
librosa
soundfile
scipy
noisereduce
pyannote.audio
python-dotenv
```

#### System Dependencies:
- **FFmpeg**: Audio/video processing
- **CUDA Toolkit**: GPU acceleration
- **cuDNN**: Deep learning GPU acceleration

### Verification

Test your installation:
```bash
# Test basic functionality
python -c "import whisper; print('Whisper OK')"
python -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')"

# Test transcription (CPU mode)
python transcribe_enhanced.py sample_audio.wav --cpu

# Test web server
python whisper_server.py
# Visit http://localhost:5555/health
```

### Troubleshooting Installation

#### Common Issues:

**1. Python version conflicts:**
```bash
# Use specific Python version
python3.9 -m venv transcribe
```

**2. CUDA not detected:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. FFmpeg not found:**
```bash
# Windows: Add to PATH or place in project folder
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

**4. cuDNN errors:**
```bash
# Use CPU fallback
python transcribe_enhanced.py audio.wav --cpu
```

**5. HuggingFace token issues:**
- Ensure token has read permissions
- Accept pyannote model license
- Check .env file format (no quotes needed)

#### Getting Help:
- Check system requirements match your setup
- Verify all dependencies are installed
- Use CPU mode if GPU issues persist
- Check GitHub issues for similar problems

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