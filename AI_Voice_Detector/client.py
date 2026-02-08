"""
Example Client Script for Voice Detection API
Demonstrates how to use the API from Python
"""

import requests
import base64
import json
import argparse
from pathlib import Path

class VoiceDetectionClient:
    """Client for interacting with Voice Detection API"""
    
    def __init__(self, api_url, api_key):
        """
        Initialize the client
        
        Args:
            api_url: Base URL of the API (e.g., http://localhost:5000)
            api_key: API authentication key
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key
        }
    
    def check_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def detect_voice(self, audio_path, language="English"):
        """
        Detect if voice is AI-generated or human
        
        Args:
            audio_path: Path to MP3 audio file
            language: Language of the audio (Tamil/English/Hindi/Malayalam/Telugu)
            
        Returns:
            dict: API response
        """
        # Validate file exists
        if not Path(audio_path).exists():
            return {"status": "error", "message": f"File not found: {audio_path}"}
        
        # Validate language
        supported_languages = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
        if language not in supported_languages:
            return {
                "status": "error",
                "message": f"Unsupported language. Use: {', '.join(supported_languages)}"
            }
        
        # Read and encode audio
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            return {"status": "error", "message": f"Failed to read audio file: {str(e)}"}
        
        # Prepare request
        payload = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
        
        # Send request
        try:
            response = requests.post(
                f"{self.api_url}/api/voice-detection",
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Could not connect to API"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def print_result(self, result):
        """Pretty print the result"""
        print("\n" + "="*70)
        print("🎙️  VOICE DETECTION RESULT")
        print("="*70)
        
        if result.get('status') == 'success':
            print(f"✅ Status: {result['status'].upper()}")
            print(f"🌐 Language: {result['language']}")
            print(f"🎯 Classification: {result['classification']}")
            print(f"📊 Confidence Score: {result['confidenceScore']:.2f} / 1.00")
            print(f"💡 Explanation: {result['explanation']}")
            
            # Interpretation
            print("\n" + "-"*70)
            if result['classification'] == 'AI_GENERATED':
                print("⚠️  This voice appears to be AI-generated or synthetic")
                if result['confidenceScore'] > 0.8:
                    print("   High confidence - Strong indicators of AI generation")
                elif result['confidenceScore'] > 0.65:
                    print("   Medium confidence - Multiple suspicious patterns detected")
                else:
                    print("   Low confidence - Some indicators present but not conclusive")
            else:
                print("✅ This voice appears to be human/real")
                if result['confidenceScore'] < 0.35:
                    print("   High confidence - Strong human characteristics")
                elif result['confidenceScore'] < 0.5:
                    print("   Medium confidence - Mostly human patterns")
                else:
                    print("   Low confidence - Close to threshold")
        else:
            print(f"❌ Status: ERROR")
            print(f"💬 Message: {result.get('message', 'Unknown error')}")
        
        print("="*70 + "\n")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Voice Detection API Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  python client.py --health

  # Detect single audio file
  python client.py --audio test_audio.mp3 --language English

  # Process multiple files
  python client.py --audio file1.mp3 --audio file2.mp3 --language Tamil

  # Use custom API URL and key
  python client.py --audio test.mp3 --url http://api.example.com --key your_api_key
        """
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:5000',
        help='API base URL (default: http://localhost:5000)'
    )
    
    parser.add_argument(
        '--key',
        default='sk_test_123456789',
        help='API key (default: sk_test_123456789)'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Check API health'
    )
    
    parser.add_argument(
        '--audio',
        action='append',
        help='Path to MP3 audio file (can be used multiple times)'
    )
    
    parser.add_argument(
        '--language',
        default='English',
        choices=['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'],
        help='Language of the audio (default: English)'
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = VoiceDetectionClient(args.url, args.key)
    
    # Health check
    if args.health:
        print("🏥 Checking API health...")
        health = client.check_health()
        print(json.dumps(health, indent=2))
        return
    
    # Process audio files
    if args.audio:
        for audio_file in args.audio:
            print(f"\n🎵 Processing: {audio_file}")
            print(f"   Language: {args.language}")
            
            result = client.detect_voice(audio_file, args.language)
            client.print_result(result)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()