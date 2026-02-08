"""
Pre-download Models Script
Downloads all required AI models before deployment
"""

import sys
import os
from pathlib import Path

print("="*70)
print("Voice Detection API - Model Download Script")
print("="*70)
print()

# Check if we're in the right directory
if not Path("requirements.txt").exists():
    print("ERROR: requirements.txt not found!")
    print("Please run this script from the project root directory.")
    sys.exit(1)

print("This script will download the following models:")
print("1. Wav2Vec2 Deepfake Detector (~1.2 GB)")
print("2. Whisper Base Language Model (~500 MB)")
print()
print("Total download size: ~1.7 GB")
print("This may take 5-15 minutes depending on your internet speed.")
print()

response = input("Continue? (y/n): ")
if response.lower() != 'y':
    print("Download cancelled.")
    sys.exit(0)

print()
print("="*70)
print("Step 1/2: Downloading Wav2Vec2 Deepfake Detector")
print("="*70)

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    
    print("Downloading model...")
    model = AutoModelForAudioClassification.from_pretrained(
        'garystafford/wav2vec2-deepfake-voice-detector'
    )
    
    print("Downloading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        'garystafford/wav2vec2-deepfake-voice-detector'
    )
    
    print("✅ Wav2Vec2 model downloaded successfully!")
    print()
    
except Exception as e:
    print(f"❌ Failed to download Wav2Vec2 model: {str(e)}")
    print("Please check your internet connection and try again.")
    sys.exit(1)

print("="*70)
print("Step 2/2: Downloading Whisper Language Detection Model")
print("="*70)

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    print("Downloading processor...")
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    
    print("Downloading model...")
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
    
    print("✅ Whisper model downloaded successfully!")
    print()
    
except Exception as e:
    print(f"❌ Failed to download Whisper model: {str(e)}")
    print("Please check your internet connection and try again.")
    sys.exit(1)

print("="*70)
print("✅ All models downloaded successfully!")
print("="*70)
print()
print("Models are cached in:", Path.home() / ".cache" / "huggingface")
print()
print("Next steps:")
print("1. The models will be automatically used by the API")
print("2. Start the API: python app.py")
print("3. Test the API: python test_api.py")
print()
print("="*70)