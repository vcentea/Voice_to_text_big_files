#!/usr/bin/env python3
"""
Test script to diagnose WhisperX diarization issues
"""

import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

def test_hf_token():
    """Test HuggingFace token availability"""
    print("\n🔍 TESTING HF_TOKEN...")
    print("=" * 40)
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN found: {hf_token[:10]}...")
        print(f"✅ Token length: {len(hf_token)} characters")
        return hf_token
    else:
        print("❌ HF_TOKEN not found!")
        print("💡 Please add your HuggingFace token to .env file:")
        print("   HF_TOKEN=your_token_here")
        return None

def test_whisperx_import():
    """Test WhisperX import and diarization pipeline"""
    print("\n🔍 TESTING WHISPERX IMPORT...")
    print("=" * 40)
    
    try:
        import whisperx
        print("✅ WhisperX imported successfully")
        
        # Check if DiarizationPipeline is available
        if hasattr(whisperx, 'diarize') and hasattr(whisperx.diarize, 'DiarizationPipeline'):
            print("✅ DiarizationPipeline available in whisperx.diarize")
            return True
        elif hasattr(whisperx, 'DiarizationPipeline'):
            print("✅ DiarizationPipeline available (legacy location)")
            return True
        else:
            print("❌ DiarizationPipeline not found in WhisperX")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import WhisperX: {e}")
        return False
    except Exception as e:
        print(f"❌ WhisperX import error: {e}")
        return False

def test_diarization_model_load(hf_token):
    """Test loading the diarization model"""
    print("\n🔍 TESTING DIARIZATION MODEL LOAD...")
    print("=" * 40)
    
    if not hf_token:
        print("❌ Cannot test model load without HF_TOKEN")
        return False
    
    try:
        import whisperx
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Using device: {device}")
        
        print("📥 Attempting to load diarization model...")
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, 
            device=device
        )
        print("✅ Diarization model loaded successfully!")
        
        # Test model type
        print(f"✅ Model type: {type(diarize_model)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load diarization model: {e}")
        print("\n💡 Common solutions:")
        print("   1. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   2. Check your HuggingFace token permissions")
        print("   3. Ensure pyannote.audio is properly installed")
        return False

def test_pyannote_import():
    """Test pyannote.audio import"""
    print("\n🔍 TESTING PYANNOTE.AUDIO IMPORT...")
    print("=" * 40)
    
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote.audio Pipeline imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pyannote.audio: {e}")
        return False
    except Exception as e:
        print(f"❌ pyannote.audio import error: {e}")
        return False

def main():
    print("🔍 WHISPERX DIARIZATION DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: HF Token
    hf_token = test_hf_token()
    
    # Test 2: PyAnnote import
    pyannote_ok = test_pyannote_import()
    
    # Test 3: WhisperX import
    whisperx_ok = test_whisperx_import()
    
    # Test 4: Model loading (only if previous tests pass)
    if hf_token and pyannote_ok and whisperx_ok:
        model_ok = test_diarization_model_load(hf_token)
    else:
        model_ok = False
        print("\n⚠️ Skipping model load test due to previous failures")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"✅ HF_TOKEN: {'OK' if hf_token else 'FAILED'}")
    print(f"✅ pyannote.audio: {'OK' if pyannote_ok else 'FAILED'}")
    print(f"✅ WhisperX: {'OK' if whisperx_ok else 'FAILED'}")
    print(f"✅ Model Load: {'OK' if model_ok else 'FAILED'}")
    
    if all([hf_token, pyannote_ok, whisperx_ok, model_ok]):
        print("\n🎉 ALL TESTS PASSED! Diarization should work.")
    else:
        print("\n❌ SOME TESTS FAILED. Diarization will be skipped.")
        
        if not hf_token:
            print("\n🔧 FIX: Add HF_TOKEN to .env file")
        if not model_ok and hf_token:
            print("\n🔧 FIX: Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")

if __name__ == "__main__":
    main() 