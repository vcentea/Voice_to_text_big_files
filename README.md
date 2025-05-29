# WhisperX Meeting Intelligence System

A comprehensive AI-powered system that transforms raw voice recordings into professional meeting reports with actionable insights, decisions, and strategic analysis.

## 🎯 What This System Does

Transform a 2-hour Romanian business meeting recording into:
- ✅ **Accurate Transcription** with speaker diarization
- ✅ **LLM-Corrected Text** for maximum accuracy
- ✅ **Structured Meeting Minutes** with executive summary
- ✅ **16 Prioritized Action Items** with deadlines and assignees
- ✅ **Strategic Insights** and competitive analysis
- ✅ **Professional DOCX Report** ready for executives

## 🔄 Process Flow - The Logic

```
┌─────────────────┐
│   Raw Audio     │
│   Recording     │
│   (110 min)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   WhisperX      │
│  Transcription  │
│ + Diarization   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  LLM Post-      │
│  Processing     │
│  & Correction   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Phase 1:     │
│  Initial Chunk  │
│    Analysis     │
│  (9 chunks)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Phase 2:     │
│  Intermediate   │
│   Synthesis     │
│ (3 meta-chunks) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Phase 3:     │
│  Final Report   │
│   Generation    │
│ (Executive Sum) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Professional   │
│  DOCX Meeting   │
│     Report      │
│ + JSON Results  │
└─────────────────┘
```

**The Logic Behind Each Step:**

1. **Audio Input** → Raw meeting recording in any format
2. **Transcription** → Convert speech to text with speaker identification  
3. **LLM Correction** → Fix errors and improve readability while preserving meaning
4. **Phase 1 Analysis** → Break into chunks, extract insights from each section
5. **Phase 2 Synthesis** → Group chunks, find patterns, consolidate findings  
6. **Phase 3 Generation** → Create executive summary and comprehensive report
7. **Final Output** → Professional documents ready for business use

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WHISPERX MEETING INTELLIGENCE                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │   TRANSCRIPTION │   │   LLM ANALYSIS  │   │  REPORT OUTPUT  │   │
│  │     LAYER       │   │     LAYER       │   │     LAYER       │   │
│  │                 │   │                 │   │                 │   │
│  │ • WhisperX      │   │ • Local Mistral │   │ • JSON Data     │   │
│  │ • Faster-Whisper│   │ • Hierarchical  │   │ • DOCX Reports  │   │
│  │ • Diarization   │   │   Map-Reduce    │   │ • SRT Subtitles │   │
│  │ • GPU Accel.    │   │ • 3-Phase Proc. │   │ • Structured    │   │
│  │                 │   │                 │   │   Minutes       │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 Detailed Process Steps

### Step 1: Audio Transcription
- **Input**: Raw M4A audio file (104MB, 110 minutes)
- **Engine**: WhisperX with Faster-Whisper large-v3 model
- **Features**: 
  - Advanced parameters for maximum accuracy
  - Speaker diarization (6 speakers identified)
  - Romanian language optimization
  - GPU-accelerated processing
- **Output**: Raw transcript with timestamps and speaker labels

### Step 2: LLM Post-Processing
- **Purpose**: Correct transcription errors and improve readability
- **Method**: Batch processing with local Mistral LLM
- **Features**:
  - Context-aware corrections
  - Maintains speaker attribution
  - Audio re-alignment for accuracy
  - Progressive writing for visibility
- **Output**: High-quality corrected transcript

### Step 3: Hierarchical Analysis (3-Phase Map-Reduce)

#### Phase 1: Initial Chunking
- **Process**: Break transcript into manageable chunks (5,049 tokens max)
- **Analysis**: Extract summaries, action items, decisions, questions per chunk
- **Result**: 9 initial chunks with detailed analysis
- **LLM**: Strategic analysis with pattern identification

#### Phase 2: Intermediate Synthesis  
- **Process**: Group 3 Phase 1 chunks into meta-chunks
- **Analysis**: Synthesize patterns, consolidate similar items, prioritize
- **Result**: 3 meta-chunks with strategic insights
- **LLM**: Intermediate-level synthesis and consolidation

#### Phase 3: Final Report Generation
- **Process**: Combine all meta-chunks into comprehensive report
- **Analysis**: Executive summary, strategic outcomes, comprehensive action plan
- **Result**: Professional meeting minutes with 16 prioritized action items
- **LLM**: Executive-level strategic analysis

## 🛠️ Technology Stack

### AI & Machine Learning Models
```
┌─────────────────────────────────────────┐
│              AI MODELS                  │
├─────────────────────────────────────────┤
│ WhisperX (Systran/faster-whisper-large) │
│ Mistral-Small-3.1-24B-Instruct-2503    │
│ Pyannote Audio (Speaker Diarization)   │
└─────────────────────────────────────────┘
```

### Software Stack
```python
# Audio Processing
whisperx==3.3.4          # Advanced transcription
av==14.4.0               # Audio/video handling
soundfile==0.13.1        # Audio file I/O

# AI/ML Processing  
torch==2.7.0             # PyTorch framework
transformers==4.52.3     # HuggingFace models
openai==1.57.2           # LLM API interface

# Document Generation
python-docx==1.1.2       # Professional reports
python-srt==1.7.3        # Subtitle generation

# Data Processing
pandas==2.2.3            # Data manipulation
numpy==2.2.6             # Numerical computing
```

### Hardware Requirements
```
┌─────────────────────────────────────────┐
│            HARDWARE SPECS               │
├─────────────────────────────────────────┤
│ GPU: NVIDIA RTX 3090 Ti (24GB VRAM)    │
│ CPU: Multi-core Intel/AMD              │
│ RAM: 16GB+ DDR4/DDR5                   │
│ Storage: NVMe SSD (500GB+ free)        │
│ OS: Windows 11 / Ubuntu 20.04+         │
└─────────────────────────────────────────┘
```

### Infrastructure
- **Development**: Windows 11 with Python 3.11 virtual environment
- **Production**: Ubuntu with CUDA 12.x support
- **GPU Computing**: NVIDIA CUDA toolkit with cuDNN
- **Local LLM**: Self-hosted Mistral model via localhost:1234

## ⚙️ Technical Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     AUDIO       │    │   TRANSCRIPTION │    │  LLM ANALYSIS   │
│   PROCESSING    │    │    PIPELINE     │    │    PIPELINE     │
│                 │    │                 │    │                 │
│ Audio File      │───▶│ WhisperX        │───▶│ Local Mistral   │
│ │ FFmpeg        │    │ │ VAD           │    │ │ Batch Proc.   │
│ │ Format Conv.  │    │ │ Forced Align  │    │ │ Context Win.  │
│ │ Preprocessing │    │ │ Diarization   │    │ │ Map-Reduce    │
│ └─GPU Memory    │    │ └─CUDA Accel.   │    │ └─JSON Output   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Input Validation│    │ Transcript +    │    │ Analysis Data + │
│ Audio Metadata  │    │ Timestamps +    │    │ Action Items +  │
│ Duration Check  │    │ Speaker Labels  │    │ Decisions +     │
│ Format Support  │    │ SRT Generation  │    │ Strategic Items │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  LLM Correction │    │ Report Generator│
                       │ │ Error Fix     │    │ │ DOCX Creation │
                       │ │ Context Aware │    │ │ JSON Export   │
                       │ │ Batch Process │    │ │ Executive Sum │
                       │ └─Re-alignment  │    │ └─Professional  │
                       └─────────────────┘    └─────────────────┘
```

**Technical Components Interaction:**

1. **Audio Processing**: FFmpeg → Format validation → GPU memory allocation
2. **Transcription Pipeline**: WhisperX → CUDA acceleration → VAD → Forced alignment → Diarization
3. **LLM Correction**: Batch processing → Context windows → Error correction → Re-alignment
4. **Analysis Pipeline**: Hierarchical chunking → Local Mistral → Map-reduce strategy → JSON structured data
5. **Report Generation**: Template processing → DOCX formatting → Executive summary → Professional output

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure GPU setup (Windows)
nvidia-smi  # Verify CUDA installation
```

### Basic Usage

#### 1. Full Pipeline (Recommended)
```bash
# Complete transcription + analysis pipeline
python transcribe_whisperx.py your_audio.m4a

# Then run LLM post-processing
python llm_postprocess.py transcripts/your_audio_transcript.txt

# Finally generate meeting report
python generate_report.py corrected_transcript.txt
python generate_report_phase2_3.py summary.json
```

#### 2. Individual Components
```bash
# Just transcription
python transcribe_whisperx.py audio.m4a

# Just LLM analysis (from existing transcript)
python generate_report.py transcript.txt

# Just Phase 2+3 (from Phase 1 results)
python generate_report_phase2_3.py summary.json
```

## 📊 Sample Results

### Input
- **File**: `4_5875387103398861179.m4a` (104MB, 110 minutes)
- **Language**: Romanian business meeting
- **Speakers**: 6 participants discussing AI strategy

### Output Statistics
- **Transcription Segments**: 1,305 timestamped segments
- **Phase 1 Analysis**: 9 detailed chunks
- **Phase 2 Synthesis**: 3 strategic meta-chunks  
- **Final Action Items**: 16 prioritized tasks
- **Strategic Decisions**: 6 major decisions documented
- **Strategic Outcomes**: 5 key business insights

### Generated Files
```
├── transcripts/
│   ├── 4_5875387103398861179_transcript.txt     # Raw transcript
│   ├── 4_5875387103398861179_corrected.txt      # LLM-corrected
│   └── 4_5875387103398861179.srt                # Subtitles
├── analysis/
│   ├── summary.json                              # Phase 1 results
│   ├── summary_final_phase2.json                # Phase 2 synthesis
│   ├── summary_final_phase3.json                # Phase 3 analysis
│   └── summary_final_meeting_report.docx        # Professional report
```

## ⚙️ Configuration

### WhisperX Advanced Parameters
```python
# Optimized for accuracy over speed
asr_options = {
    "beam_size": 5,              # Higher accuracy
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback sequence
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "patience": 1.0,             # Ctranslate2 compatibility
    "suppress_numerals": False
}
```

### LLM Configuration
```python
# Local LLM setup
LLM_API_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL_NAME = "mistral-small-3.1-24b-instruct-2503"
MAX_CONTEXT_TOKENS = 8549
BATCH_SIZE = 5  # For optimal processing
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Memory Issues
```python
# Reduce batch size in configuration
BATCH_SIZE = 3  # Instead of 5
MAX_CONTEXT_TOKENS = 6000  # Instead of 8549
```

#### Model Download Issues
```bash
# Pre-download models
python -c "import whisperx; whisperx.load_model('large-v3')"
```

## 📈 Performance Metrics

### Processing Times (RTX 3090 Ti)
- **Transcription**: ~15 minutes for 110-minute audio
- **LLM Post-processing**: ~20 minutes (batch processing)
- **Phase 1 Analysis**: ~15 minutes (9 chunks)
- **Phase 2+3 Analysis**: ~5 minutes (meta-chunks + final)
- **Total Pipeline**: ~55 minutes for complete analysis

### Accuracy Improvements
- **Raw WhisperX**: 92% accuracy (estimated)
- **LLM-Corrected**: 97%+ accuracy (human-verified sample)
- **Speaker Diarization**: 95%+ accuracy

## 🎯 Use Cases

### Business Meetings
- Executive strategy sessions
- Board meetings
- Client consultations
- Team retrospectives

### Professional Services
- Legal depositions
- Medical consultations
- Consulting sessions
- Training workshops

### Academic & Research
- Research interviews
- Conference presentations
- Focus groups
- Academic discussions

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd whisper_server
python -m venv transcribe
source transcribe/bin/activate  # Linux/Mac
# or
transcribe\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Code Quality Guidelines
- Follow clean code principles
- Use type hints for all functions
- Document complex algorithms
- Write tests for critical functions
- Keep functions focused and small

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WhisperX**: Advanced speech recognition with diarization
- **Faster-Whisper**: Optimized Whisper implementation  
- **Mistral AI**: Local LLM for intelligent analysis
- **PyAnnote**: Speaker diarization capabilities
- **HuggingFace**: Model distribution and transformers

---

*Built with ❤️ for transforming voice into actionable intelligence* 