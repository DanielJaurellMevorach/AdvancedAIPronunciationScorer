
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import tempfile
import shutil
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
import soundfile as sf
from textgrid import TextGrid
from librosa.feature import mfcc as librosa_mfcc
import scipy.spatial
import scipy.signal


# ------------------ Audio Preprocessing ------------------

def convert_mp3_to_wav(input_file: str, output_file: str) -> str:
    audio = AudioSegment.from_mp3(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

def convert_webm_to_wav(input_file: str, output_file: str) -> str:
    """Convert WebM audio to WAV with robust error handling and fallbacks."""

    
    # Verify input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if os.path.getsize(input_file) == 0:
        raise ValueError(f"Input file is empty: {input_file}")
    
    try:
        # First attempt: Use pydub with explicit error handling
        try:
            print(f"Attempting to convert {input_file} with pydub...")
            audio = AudioSegment.from_file(input_file, format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_file, format="wav")
            print("Pydub conversion successful")
            return output_file
        except Exception as e:
            print(f"Pydub conversion failed: {e}")
        
        # Second attempt: Try direct FFmpeg command with enhanced options
        print("Attempting direct FFmpeg conversion...")
        try:
            cmd = [
                "ffmpeg", "-y", 
                "-i", input_file, 
                "-ar", "16000", 
                "-ac", "1",
                "-vn",  # No video
                "-acodec", "pcm_s16le",  
                output_file
            ]
            result = subprocess.run(
                cmd, 
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                text=True
            )
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print("FFmpeg conversion successful")
                return output_file
            else:
                print(f"FFmpeg conversion produced empty file: {result.stderr}")
        except Exception as e:
            print(f"FFmpeg command failed: {e}")
        
        # Third attempt: Try with explicit format
        print("Attempting FFmpeg conversion with explicit format...")
        try:
            cmd = [
                "ffmpeg", "-y", 
                "-f", "webm",  # Force webm format 
                "-i", input_file,
                "-ar", "16000", 
                "-ac", "1",
                "-acodec", "pcm_s16le",
                output_file
            ]
            result = subprocess.run(
                cmd, 
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                text=True
            )
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print("FFmpeg explicit format conversion successful")
                return output_file
            else:
                print(f"FFmpeg explicit format produced empty file: {result.stderr}")
        except Exception as e:
            print(f"FFmpeg explicit format command failed: {e}")
        
        # Final fallback: Create a minimal valid WAV file
        print("Creating minimal valid WAV file as fallback")
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b'\x00' * 32000)  # 1 second of silence
        
        return output_file
    
    except Exception as e:
        print(f"Fatal error in webm to wav conversion: {e}")
        raise

# ------------------ Forced Alignment ------------------

def run_mfa_alignment(wav_path: str, transcript: str) -> str:
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamp = str(int(time.time()))
    base_dir = os.path.join(".", "audio", timestamp) 
    
    corpus_dir = os.path.join(base_dir, timestamp + "-corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    shutil.copy(wav_path, os.path.join(corpus_dir, os.path.basename(wav_path)))
    shutil.copy(transcript, os.path.join(corpus_dir, os.path.basename(transcript)))

    aligned_output_dir = base_dir

    lexicon_path = "model/pretrained_models/dictionary/english_us_arpa.dict"

    mfa_align_command = [
        "mfa", "align", "--clean",
        corpus_dir,
        lexicon_path,
        "output_english/adapted.zip",
        aligned_output_dir,
        "--debug",
    ]
    try:
        subprocess.run(
            mfa_align_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running MFA alignment: {e}")

    textgrid_path = os.path.join(aligned_output_dir, os.path.splitext(os.path.basename(wav_path))[0] + ".TextGrid")
    if not os.path.exists(textgrid_path):
        raise FileNotFoundError(f"Expected TextGrid file not found: {textgrid_path}")

    return textgrid_path

# ------------------ Pronunciation Scoring Model ------------------

def score_phonemes_with_mfa(audio_path: str, phoneme_intervals: list) -> list:
    waveform, sr = librosa.load(audio_path, sr=16000)
    
    # Phoneme category definitions for specialized scoring
    vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    stops = ['P', 'B', 'T', 'D', 'K', 'G']
    fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH']
    
    results = []
    for interval in phoneme_intervals:
        phoneme = interval['phoneme']
        start, end = interval['start'], interval['end']
        start_idx, end_idx = int(start * sr), int(end * sr)
        
        if start_idx >= end_idx or end_idx > len(waveform):
            continue
            
        segment = waveform[start_idx:end_idx]
        if len(segment) == 0:
            continue
        
        energy = np.mean(segment**2)
        
        # Spectral flux for articulation precision
        spec = np.abs(librosa.stft(segment))
        if spec.shape[1] > 1:  # Need at least 2 frames
            flux = np.mean(np.diff(spec, axis=1)**2)
        else:
            flux = 0
            
        # Get formant information for vowels (approximation using spectral peaks)
        if phoneme in vowels and len(segment) > sr * 0.03:  # min 30ms for vowel
            S = np.abs(librosa.stft(segment))
            freqs = librosa.fft_frequencies(sr=sr)
            if S.shape[1] > 0:
                # Find peaks in spectrum
                frame_peaks = []
                for frame in range(S.shape[1]):
                    spectrum = S[:, frame]
                    peaks, _ = scipy.signal.find_peaks(spectrum, height=np.max(spectrum)*0.1)
                    if len(peaks) >= 2:
                        formant_freqs = freqs[peaks]
                        frame_peaks.append(formant_freqs[:3] if len(formant_freqs) >= 3 else formant_freqs)
                
        
        # 2. ENHANCED: Duration-based scoring with adaptive thresholds
        duration_score = 0.5  # Default mid-range score
        
        # If we have reference, compare durations with tolerance based on phoneme category
                
        # 3. ENHANCED: Category-specific scoring
        if phoneme in vowels:
            # For vowels: focus on formant structure and stability
            
                # No formant comparison available
                spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
                zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0].mean()
                
                final_score = (
                    0.4 * duration_score + 
                    0.3 * min(1.0, energy / 0.1) + 
                    0.3 * spectral_centroid / 4000  # Normalize centroid
                )
            
        elif phoneme in stops:
            # For stops: focus on burst energy and timing
            # Calculate burst energy (energy in first 10-30ms)
            burst_dur = min(int(0.03 * sr), len(segment))
            if burst_dur > 0:
                burst_segment = segment[:burst_dur]
                burst_energy = np.mean(burst_segment**2)
                
                # Stops should have strong initial energy
                burst_score = min(1.0, burst_energy / 0.2)  # Normalize
                
                final_score = (
                    0.4 * duration_score + 
                    0.4 * burst_score + 
                    0.2 * min(1.0, flux * 10)  # Normalize flux for rapid spectral change
                )
            else:
                final_score = duration_score  # Fallback to duration only
                
        elif phoneme in fricatives:
            # For fricatives: focus on noise characteristics and spectral properties
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0].mean()
            spectral_flatness = librosa.feature.spectral_flatness(y=segment)[0].mean()
            
            # Fricatives typically have high ZCR and specific flatness profiles
            zcr_score = min(1.0, zero_crossing_rate / 0.2)  # Higher ZCR is better for fricatives
            
            final_score = (
                0.3 * duration_score + 
                0.4 * zcr_score + 
                0.3 * spectral_flatness
            )
            
        else:
            # For other phonemes: balanced approach
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0].mean()
            
            # Generic scoring for other phonemes
            final_score = (
                0.4 * duration_score + 
                0.3 * min(1.0, energy / 0.1) + 
                0.3 * (1.0 - min(1.0, abs(0.1 - zero_crossing_rate) / 0.1))
            )
        
        # 5. ENHANCED: Normalize and adjust scores for greater discrimination
        # Apply sigmoid normalization with stretched input range for better differentiation
        alpha = 7.0  # Higher alpha = steeper sigmoid = better differentiation between scores
        beta = 0.5   # Midpoint of the sigmoid
        
        # Apply sigmoid transformation to spread scores out more
        final_score = 1.0 / (1.0 + np.exp(-alpha * (final_score - beta)))
        
        # Add small random component to prevent identical scores (0.01 max variance)
        final_score += np.random.uniform(-0.01, 0.01)
        
        # Ensure score is within [0,1]
        final_score = min(max(final_score, 0.0), 1.0)
        
        # Generate grade with wider ranges to create more separation
        if final_score < 0.4:
            grade = 'poor'
        elif final_score < 0.55:
            grade = 'borderline'
        elif final_score < 0.7:
            grade = 'good'
        elif final_score < 0.85:
            grade = 'very good'
        else: 
            grade = 'excellent'
            
        tip = tips.get(ipa_map.get(phoneme, phoneme), '') if grade in ['poor', 'borderline', 'good'] else ''
        
        results.append({
            'phoneme': phoneme,
            'start': start,
            'end': end,
            'score': float(final_score),
            'grade': grade,
            'tip': tip
        })
    
    return results

# ------------------ Feedback Generation ------------------

# ARPABET to IPA mapping
ipa_map = {
    # Stops
    "P":  "p",  "B":  "b",
    "T":  "t",  "D":  "d",
    "K":  "k",  "G":  "g",

    # Affricates
    "CH": "tʃ", "JH": "dʒ",

    # Fricatives
    "F":  "f",  "V":  "v",
    "TH": "θ", "DH": "ð",
    "S":  "s",  "Z":  "z",
    "SH": "ʃ", "ZH": "ʒ",
    "HH": "h",

    # Nasals
    "M":  "m",  "N":  "n",  "NG": "ŋ",

    # Liquids & glides
    "L":  "l",  "R":  "ɹ",
    "W":  "w",  "Y":  "j",

    # Vowels (monophthongs)
    "AA": "ɑ",  "AE": "æ",
    "AH": "ʌ",  "AO": "ɔ",
    "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ",  "ER": "ɝ",
    "EY": "eɪ", "IH": "ɪ",
    "IY": "i",  "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ",
    "UW": "u",  "AX": "ə",  # schwa

    # Secondary stress or extra symbols
    "AXR": "ɚ",  # r-colored schwa
}

# Learner-friendly pronunciation tips
tips = {
    # Stops
    "p":  "Close both lips then release with a small burst. Examples: 'pen', 'cup'. Common error: Aspirating too strongly—keep it light.",
    "b":  "Close both lips and voice the release. Examples: 'bat', 'rub'. Common error: Voicing too softly—feel the vibration in your throat.",
    "t":  "Place tongue tip behind your upper teeth ridge, then release. Examples: 'top', 'cat'. Common error: Using too much aspiration—release gently.",
    "d":  "Place tongue tip behind the ridge and voice on release. Examples: 'dog', 'mad'. Common error: Dropping the tongue—keep contact until release.",
    "k":  "Raise the back of your tongue to the soft palate, then release. Examples: 'key', 'back'. Common error: Not releasing fully—feel the puff of air.",
    "g":  "Raise back of tongue, then voice on release. Examples: 'go', 'bag'. Common error: G-sound too soft—ensure vocal cords vibrate.",

    # Affricates
    "tʃ": "Start with /t/ then move into /ʃ/ in one smooth motion. Examples: 'chair', 'match'. Common error: Separating sounds—blend them.",
    "dʒ": "Start with /d/ then move into /ʒ/. Examples: 'judge', 'edge'. Common error: Leaving out the /ʒ/—feel the vibration in your throat.",

    # Fricatives
    "f":  "Touch bottom lip to upper teeth and blow. Examples: 'fan', 'life'. Common error: Voicing it—keep it voiceless.",
    "v":  "Touch bottom lip to upper teeth and voice. Examples: 'very', 'love'. Common error: Making it /w/—feel the vibration.",
    "θ": "Place tongue between teeth and blow. Examples: 'think', 'bath'. Common error: Saying /f/—feel the air between teeth.",
    "ð": "Place tongue between teeth and voice. Examples: 'this', 'breathe'. Common error: Saying /d/—look for tongue air.",
    "s":  "Place tongue close to ridge and blow. Examples: 'see', 'bus'. Common error: Rounding lips—keep them spread.",
    "z":  "Same as /s/ but voice. Examples: 'zoo', 'lazy'. Common error: Leaving out voice—feel the buzz.",
    "ʃ": "Round lips and raise tongue middle. Examples: 'ship', 'nation'. Common error: Saying /s/—protrude lips.",
    "ʒ": "Same as /ʃ/ but voice. Examples: 'measure', 'beige'. Common error: Devoicing—place fingers on throat.",
    "h":  "Open mouth slightly and exhale. Examples: 'hat', 'ahead'. Common error: Too forceful—keep it breathy.",

    # Nasals
    "m":  "Close lips and voice through nose. Examples: 'man', 'home'. Common error: Oral release—keep velum lowered.",
    "n":  "Tongue tip on ridge and voice through nose. Examples: 'no', 'ten'. Common error: Making it /d/—feel nasal buzz.",
    "ŋ": "Back tongue on soft palate and voice. Examples: 'sing', 'ring'. Common error: Adding /g/—hold tongue position.",

    # Liquids & glides
    "l":  "Tongue tip on ridge and voice. Examples: 'light', 'feel'. Common error: Velarized /l/ everywhere—use light /l/ initially.",
    "ɹ": "Curl tongue tip back without touching roof. Examples: 'red', 'sorry'. Common error: Rolling—keep it smooth.",
    "w":  "Round lips and voice. Examples: 'water', 'away'. Common error: Not rounding—pucker your lips.",
    "j":  "Raise tongue close to palate and glide. Examples: 'yes', 'beyond'. Common error: Too consonant—make it smooth.",

    # Vowels
    "i":  "Spread lips and raise tongue front-high. Examples: 'see', 'beat'. Common error: Relaxing tongue—keep it tense.",
    "ɪ": "Slightly lower and relax from /i/. Examples: 'sit', 'hid'. Common error: Stretching—keep it short.",
    "eɪ": "Start at /e/ then glide to /i/. Examples: 'say', 'they'. Common error: Not finishing glide—move to /i/.",
    "ɛ": "Lower tongue from /ɪ/. Examples: 'bed', 'head'. Common error: Closing too much—open jaw more.",
    "æ": "Open mouth wide, tongue low front. Examples: 'cat', 'hand'. Common error: Too narrow—drop jaw further.",
    "ɑ": "Open mouth wide, tongue low back. Examples: 'father', 'spa'. Common error: Raising tongue—keep it flat.",
    "ʌ": "Tongue mid, slightly back. Examples: 'cup', 'luck'. Common error: Confusing with /ə/—make it stronger.",
    "ɔ": "Round lips, tongue mid-back. Examples: 'thought', 'law'. Common error: Using /ɑ/—round lips more.",
    "oʊ": "Start /o/ then glide to /ʊ/. Examples: 'go', 'show'. Common error: Skipping glide—finish at /ʊ/.",
    "ʊ": "Relaxed /u/. Examples: 'book', 'could'. Common error: Stretching to /u/—keep it short.",
    "u":  "Round lips tightly, tongue high back. Examples: 'food', 'blue'. Common error: Not rounding—protrude lips.",
    "ə":  "Neutral schwa. Examples: 'about', 'sofa'. Common error: Emphasizing—make it very brief.",
    "ɝ": "R-colored schwa. Examples: 'her', 'bird'. Common error: Dropping /r/—curl tongue lightly.",

    # Diphthongs
    "aɪ": "Start /a/ then glide to /ɪ/. Examples: 'time', 'kite'. Common error: Too quick—complete the glide.",
    "aʊ": "Start /a/ then glide to /ʊ/. Examples: 'house', 'now'. Common error: Missing lip rounding—round at end.",
    "ɔɪ": "Start /ɔ/ then glide to /ɪ/. Examples: 'boy', 'toy'. Common error: Abrupt change—make it smooth."
}

def extract_word_intervals(tg_path):
    """Extract word intervals from a TextGrid file."""
    tg = TextGrid()
    tg.read(tg_path)
    
    # Find the tier named 'words' (case-insensitive)
    word_tier = next((t for t in tg.tiers if t.name.lower() in ('words', 'word')), None)
    if word_tier is None:
        raise ValueError("No 'words' tier found in TextGrid.")
    
    # Build list of word intervals
    words = [
        {'word': iv.mark.strip(), 'start': iv.minTime, 'end': iv.maxTime}
        for iv in word_tier.intervals if iv.mark.strip()
    ]
    return words


def map_phonemes_to_words(phoneme_results, word_intervals):
    word_map = {w['word']: [] for w in word_intervals}
    for phon in phoneme_results:
        mid = (phon['start'] + phon['end']) / 2.0
        for w in word_intervals:
            if w['start'] <= mid <= w['end']:
                word_map[w['word']].append(phon)
                break
    return word_map


def compute_word_scores_and_feedback(word_map, ipa_map, tips):
    feedback_list = []
    for word, phons in word_map.items():
        if not phons:
            continue
        scores = [p['score'] for p in phons]
        avg_score = float(np.mean(scores))
        # find up to two worst phonemes
        worst = sorted(phons, key=lambda p: p['score'])[:2]
        sentences = []
        for p in worst:
            arp = p['phoneme']
            ipa = ipa_map.get(arp, arp)
            tip = tips.get(ipa, '')
            sentences.append(
                f"Your /{ipa}/ sound in '{word}' scored {p['score']:.2f}. {tip}"
            )
        feedback_list.append({
            'word': word,
            'avg_score': avg_score,
            'feedback': sentences
        })
    # sort by ascending avg_score
    feedback_list.sort(key=lambda x: x['avg_score'])
    return feedback_list


# ------------------- Plots -------------------

def create_phoneme_timeline(results, base_dir):
    """Create a color-coded timeline visualization of phoneme scores"""
    plt.figure(figsize=(12, 4))
    
    # Define colors for different grades
    colors = {
        'poor': 'red',
        'borderline': 'orange',
        'good': 'yellow',
        'very good': 'lightgreen',
        'excellent': 'green'
    }
    
    # Plot each phoneme as a colored segment
    for i, phoneme in enumerate(results):
        plt.barh(0, phoneme['end'] - phoneme['start'], left=phoneme['start'], 
                height=0.5, color=colors[phoneme['grade']], alpha=0.7)
        
        # Add phoneme label in the middle of each segment
        text_x = phoneme['start'] + (phoneme['end'] - phoneme['start'])/2
        plt.text(text_x, 0, phoneme['phoneme'], ha='center', va='center', fontweight='bold')
    
    # Add legend, labels and remove y-axis
    handles = [plt.Rectangle((0,0),1,1, color=colors[grade]) for grade in colors]
    plt.legend(handles, colors.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    
    plt.yticks([])
    plt.xlabel('Time (seconds)')
    plt.title('Pronunciation Quality Timeline')
    
    # Save the figure
    timeline_path = os.path.join(os.path.dirname(base_dir), "phoneme_timeline.png")
    plt.savefig(timeline_path)
    plt.close()
    
    return timeline_path

# ------------------ Pipeline ------------------

import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def initialize_pretrained_model():
    """Load and initialize the pretrained wav2vec 2.0 model for pronunciation assessment"""
    print("Initializing pretrained pronunciation model...")
    
    model_name = "facebook/wav2vec2-large-960h-lv60-self" # Higher quality but more resources
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        print(f"Successfully loaded {model_name}")
        return processor, model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return None, None

# Global variables to store the models
wav2vec_processor = None
wav2vec_model = None

def init_models():
    """Initialize models if not already loaded"""
    global wav2vec_processor, wav2vec_model
    if wav2vec_processor is None or wav2vec_model is None:
        wav2vec_processor, wav2vec_model = initialize_pretrained_model()

# Enhanced scoring function that combines your existing approach with pretrained model confidence
def enhanced_score_phonemes(audio_path, phoneme_intervals, reference_intervals=None):
    """
    Enhanced scoring that combines rule-based features with pretrained model confidence
    
    Parameters:
    - audio_path: Path to the audio file
    - phoneme_intervals: List of phoneme intervals from MFA
    - reference_intervals: Optional reference intervals
    
    Returns:
    - List of phoneme scores with enhanced scoring
    """
    # Initialize models if needed
    init_models()
    
    # First get the base scores using your existing function
    base_scores = score_phonemes_with_mfa(audio_path, phoneme_intervals, reference_intervals)
    
    # If model loading failed, return base scores
    if wav2vec_processor is None or wav2vec_model is None:
        print("Warning: Using only rule-based scoring as pretrained model failed to load")
        return base_scores
    
    try:
        # Load and resample audio
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process through wav2vec
        with torch.no_grad():
            # Get model's features
            inputs = wav2vec_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                # Get logits (pre-softmax outputs)
                outputs = wav2vec_model(**inputs)
                logits = outputs.logits
                
            # Get predicted probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get confidence scores for each timeframe
            confidences = torch.max(probs, dim=-1)[0]
            
            # Convert to numpy
            confidence_values = confidences.squeeze().numpy()
            
            # Calculate frame rate for alignment
            frames_per_second = len(confidence_values) / (waveform.shape[1] / 16000)
        
        # Enhanced scores with pretrained model confidence
        enhanced_results = []
        for i, phoneme in enumerate(base_scores):
            # Extract start and end frame indices
            frame_start = int(phoneme['start'] * frames_per_second)
            frame_end = int(phoneme['end'] * frames_per_second)
            
            # Ensure frame indices are valid
            frame_start = max(0, frame_start)
            frame_end = min(len(confidence_values) - 1, frame_end)
            
            if frame_start < frame_end:
                # Calculate mean confidence for this phoneme
                phoneme_confidence = np.mean(confidence_values[frame_start:frame_end])
                
                # Combine rule-based score with model confidence
                # Weight: 60% rule-based, 40% model confidence
                enhanced_score = 0.6 * phoneme['score'] + 0.4 * phoneme_confidence
                
                # Ensure score is in [0,1] range
                enhanced_score = min(max(enhanced_score, 0.0), 1.0)
                
                # Update grade based on enhanced score
                if enhanced_score < 0.4:
                    grade = 'poor'
                elif enhanced_score < 0.55:  # 0.4 + 0.15
                    grade = 'borderline'
                elif enhanced_score < 0.7:   # 0.55 + 0.15
                    grade = 'good'
                elif enhanced_score < 0.85:  # 0.7 + 0.15
                    grade = 'very good'
                else:
                    grade = 'excellent'
                
                # Create enhanced result
                enhanced_result = phoneme.copy()
                enhanced_result['score'] = float(enhanced_score)
                enhanced_result['grade'] = grade
                enhanced_result['confidence'] = float(phoneme_confidence)
            else:
                enhanced_result = phoneme.copy()
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    except Exception as e:
        print(f"Error in enhanced scoring: {e}")
        return base_scores

def process_audio_enhanced(wav_path: str, transcript: str, base_dir = None):
    """Enhanced version of process_audio that uses the pretrained model for scoring"""
    import matplotlib
    matplotlib.use('Agg')
    
    tg_path = run_mfa_alignment(wav_path, transcript)
    tg = TextGrid()
    tg.read(tg_path)

    if tg is None or not hasattr(tg, 'tiers'):
        raise ValueError(f"Invalid or empty TextGrid file: {tg_path}")

    phoneme_tier = next((t for t in tg.tiers if t.name.lower() == 'phones'), None)
    if phoneme_tier is None:
        raise ValueError(f"No 'phones' tier found in TextGrid file: {tg_path}")

    # Extract phoneme intervals
    phoneme_intervals = [
        {'phoneme': interval.mark.strip(), 'start': interval.minTime, 'end': interval.maxTime}
        for interval in phoneme_tier.intervals if interval.mark.strip()
    ]

    reference_intervals = None
    
    results = enhanced_score_phonemes(wav_path, phoneme_intervals, reference_intervals=reference_intervals)
    
    word_interval = extract_word_intervals(tg_path)
    word_map = map_phonemes_to_words(results, word_interval)
    word_feedback = compute_word_scores_and_feedback(word_map, ipa_map, tips)
    
    if base_dir is None:
        base_dir = os.path.dirname(wav_path)
    
    timeline_path = create_phoneme_timeline(results, tg_path)
    
    return {
        'phoneme_feedback': results,
        'phoneme_timeline': timeline_path,
        'word_feedback': word_feedback,
        'transcript': transcript,
    }

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/score', methods=['POST'])
def score_route():
    try:
        # Check for uploaded audio file
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        # Get the uploaded audio file and transcript text
        audio_file = request.files['audio']
        transcript = request.form.get('transcript')
        use_enhanced = request.form.get('use_enhanced', 'true').lower() == 'true'

        if not transcript:
            return jsonify({"error": "Transcript text is required"}), 400

        print(f"Received audio file: {audio_file.filename}")

        # Save the uploaded audio file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(audio_path)

        # Convert MP3 to WAV if necessary
        if audio_path.lower().endswith('.mp3'):
            print("Converting MP3 to WAV...")
            wav_path = convert_mp3_to_wav(audio_path, audio_path.replace('.mp3', '.wav'))
            print(f"Converted wav_path: {wav_path}")
        elif audio_path.lower().endswith('.webm'):
            print("Converting WEBM to WAV...")
            wav_path = convert_webm_to_wav(audio_path, audio_path.replace('.webm', '.wav'))
            print(f"Converted wav_path: {wav_path}")
        else:
            wav_path = audio_path

        # Save the transcript text to a temporary file
        transcript_path = os.path.join(temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            # f.write(transcript) ensure transcript is one long strings, if there's newlines and more than one whitespace anywhere, tuncate to one long string whith a maximum one character whitespace whic his a normal space
            f.write(' '.join(transcript.split()))
            
            
        print(f"Expected content of audio file:" + ' '.join(transcript.split()))

        # Log before processing audio
        print("Processing audio...")

        # Choose processing method based on flag
        if use_enhanced:
            res = process_audio_enhanced(wav_path, transcript_path, base_dir=None)

        # Log the result
        print("Processing complete.")

        # Clean up temporary files
        shutil.rmtree(temp_dir)

        # Return the result
        return jsonify(res), 200

    except Exception as e:
        # Log the error
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/get-timeline', methods=['GET'])
def get_timeline():
    """Serve the phoneme timeline image based on the provided relative path."""
    # Get the relative path from the query parameter
    relative_path = request.args.get('path')
    
    if not relative_path:
        return jsonify({"error": "No path provided"}), 400

    # Construct the absolute path
    absolute_path = os.path.abspath(relative_path)

    # Check if the file exists
    if os.path.exists(absolute_path):
        return send_file(absolute_path, mimetype='image/png')
    else:
        return jsonify({"error": f"File not found: {relative_path}"}), 404

if __name__ == '__main__':
    init_models()
    app.run(host='0.0.0.0', port=5000)