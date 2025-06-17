# --- IMPORTS ---
# Standard library imports
import json
import base64
import random
import string
import os
import abc

# Third-party imports
import soundfile as sf
import torch
import numpy as np
import pandas as pd
from torchaudio.transforms import Resample
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from dtwalign import dtw_from_distance_matrix
import epitran
import eng_to_ipa

# --- FLASK WEB APP INITIALIZATION ---
app = Flask(__name__, template_folder='.')
CORS(app)

# --- DATA LOADING ---
class TextDataset:
    """Simple class to hold sentences from a CSV file."""
    def __init__(self, filepath):
        self.table_dataframe = pd.read_csv(filepath, delimiter=';')
        self.number_of_samples = len(self.table_dataframe)

    def get_random_sample(self):
        """Returns a random sentence from the dataset."""
        idx = random.randint(0, self.number_of_samples - 1)
        return self.table_dataframe['sentence'].iloc[idx]

# --- MODEL INTERFACES ---
class IASRModel(metaclass=abc.ABCMeta):
    """Abstract Base Class for Automatic Speech Recognition models."""
    @abc.abstractmethod
    def getTranscript(self) -> str:
        raise NotImplementedError
    @abc.abstractmethod
    def getWordLocations(self) -> list:
        raise NotImplementedError
    @abc.abstractmethod
    def processAudio(self, audio):
        raise NotImplementedError

class ITextToPhonemModel(metaclass=abc.ABCMeta):
    """Abstract Base Class for Text-to-Phoneme models."""
    @abc.abstractmethod
    def convertToPhonem(self, str) -> str:
        raise NotImplementedError

# --- WHISPER ASR MODEL ---
class WhisperASRModel(IASRModel):
    """Wrapper for Whisper ASR model from Hugging Face transformers."""
    def __init__(self, model_name='openai/whisper-base'):
        self.asr = pipeline(
            'automatic-speech-recognition',
            model=model_name,
            device=torch.device('cpu'),
            model_kwargs={"attn_implementation": "eager"}
        )
        self._transcript = ''
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio: np.ndarray):
        """Processes audio and extracts transcript (fallback without timestamps due to tensor bug)."""
        try:
            # Just get transcript without timestamps to avoid the tensor shape bug
            result = self.asr(audio)
            self._transcript = result['text']
            self._word_locations = []  # Empty for now due to Whisper bug
        except Exception as e:
            print(f"Error in Whisper processing: {e}")
            self._transcript = ""
            self._word_locations = []

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        return self._word_locations

# --- PHONEME CONVERTERS ---
class EpitranPhonemConverter(ITextToPhonemModel):
    """Converts French text to phonemes using the 'epitran' library."""
    def __init__(self, epitran_model) -> None:
        self.epitran_model = epitran_model
    def convertToPhonem(self, sentence: str) -> str:
        return self.epitran_model.transliterate(sentence)

class EngPhonemConverter(ITextToPhonemModel):
    """Converts English text to phonemes using the 'eng-to-ipa' library."""
    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        return phonem_representation.replace('*', '')

# --- PRONUNCIATION TRAINER ---
class PronunciationTrainer:
    """Main class that orchestrates the pronunciation scoring process."""
    def __init__(self, asr_model: IASRModel, phonem_converter: ITextToPhonemModel, sampling_rate: int = 16000):
        self.asr_model = asr_model
        self.ipa_converter = phonem_converter
        self.sampling_rate = sampling_rate
        self.categories_thresholds = np.array([80, 50, 0])

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalizes audio to zero mean and scales to [-1, 1]."""
        if audio.any():
            audio = audio - np.mean(audio)
            audio = audio / np.max(np.abs(audio))
        return audio

    def _get_transcript_and_locations(self, audio: np.ndarray):
        """Processes audio to get transcript using ASR model."""
        self.asr_model.processAudio(audio)
        transcript = self.asr_model.getTranscript()
        word_locations = self.asr_model.getWordLocations()
        return transcript, word_locations

    def _remove_punctuation(self, text: str) -> str:
        """Removes all punctuation from text."""
        return ''.join(char for char in text if char not in string.punctuation)

    def get_pronunciation_accuracy(self, real_and_transcribed_words: list) -> tuple:
        """Calculates pronunciation accuracy using IPA phoneme comparison."""
        total_distance = 0
        total_phonemes = 0
        word_accuracies = []
        
        # Track how many words were actually transcribed vs missing
        transcribed_words = 0
        total_words = 0

        for real_word, trans_word in real_and_transcribed_words:
            real_word_clean = self._remove_punctuation(real_word).lower()
            trans_word_clean = self._remove_punctuation(trans_word).lower() if trans_word != '-' else ''

            total_words += 1

            # Skip empty real words (punctuation only)
            if not real_word_clean:
                word_accuracies.append(100)  # Punctuation gets full credit
                continue

            real_ipa = self.ipa_converter.convertToPhonem(real_word_clean)
            
            if not real_ipa: 
                word_accuracies.append(0)
                continue

            # If no transcribed word, score based on context
            if not trans_word_clean:
                # Give partial credit if most other words were transcribed correctly
                word_accuracies.append(0)  # Will be adjusted later
                total_distance += len(real_ipa)
                total_phonemes += len(real_ipa)
            else:
                transcribed_words += 1
                trans_ipa = self.ipa_converter.convertToPhonem(trans_word_clean)
                
                if not trans_ipa:
                    distance = len(real_ipa)
                else:
                    distance = edit_distance(real_ipa, trans_ipa)
                
                num_phonemes = len(real_ipa)
                total_distance += distance
                total_phonemes += num_phonemes
                
                # Calculate word accuracy
                accuracy = ((num_phonemes - distance) / num_phonemes) * 100 if num_phonemes > 0 else 0
                word_accuracies.append(max(0, accuracy))

        # Calculate overall accuracy with context-aware scoring
        if total_phonemes > 0:
            overall_accuracy = ((total_phonemes - total_distance) / total_phonemes) * 100
            overall_accuracy = max(0, overall_accuracy)
            
            # If user got most words right, don't penalize too heavily for a few mumbled words
            transcription_rate = transcribed_words / max(total_words, 1)
            if transcription_rate > 0.6:  # If more than 60% of words were transcribed
                # Boost the score to reflect that most pronunciation was good
                overall_accuracy = min(100, overall_accuracy * (1 + transcription_rate * 0.3))
                
                # Also boost individual word scores for missing words in good contexts
                for i, (real_word, trans_word) in enumerate(real_and_transcribed_words):
                    if trans_word == '-' and word_accuracies[i] == 0:
                        # Give some partial credit for missing words when overall performance is good
                        word_accuracies[i] = min(30, overall_accuracy * 0.3)
            
        else:
            overall_accuracy = 0

        return (np.round(max(0, overall_accuracy)), word_accuracies)

    def get_words_pronunciation_category(self, accuracies: list) -> list:
        """Converts accuracy scores to categories (1=Good, 2=Medium, 3=Poor)."""
        categories = []
        for accuracy in accuracies:
            category_index = np.argmin(np.abs(self.categories_thresholds - accuracy))
            categories.append(category_index + 1)
        return categories

    def process_audio_for_text(self, recorded_audio: np.ndarray, real_text: str) -> dict:
        """Main method that processes audio and returns pronunciation analysis."""
        # 1. Preprocess audio and get transcript
        processed_audio = self._preprocess_audio(recorded_audio)
        recording_transcript, _ = self._get_transcript_and_locations(processed_audio)
        
        # 2. Align words using DTW
        words_real = real_text.split()
        words_estimated = recording_transcript.split()
        mapped_words, _ = get_best_mapped_words(words_estimated, words_real)
        
        real_and_transcribed_words = list(zip(words_real, mapped_words))

        # 3. Calculate pronunciation accuracy using IPA phoneme comparison
        pronunciation_accuracy, word_accuracies = self.get_pronunciation_accuracy(real_and_transcribed_words)
        pronunciation_categories = self.get_words_pronunciation_category(word_accuracies)
        
        # 4. Letter-level correctness for highlighting
        is_letter_correct_all_words = ''
        for idx, real_word in enumerate(words_real):
            transcribed_word = mapped_words[idx]
            is_letter_correct = get_which_letters_were_correct(real_word, transcribed_word)
            is_letter_correct_all_words += ''.join(map(str, is_letter_correct)) + ' '
            
        # 5. Return results
        result = {
            'real_transcript': real_text,
            'recording_transcript': recording_transcript,
            'matched_transcripts': ' '.join(mapped_words),
            'pronunciation_accuracy': str(int(pronunciation_accuracy)),
            'pair_accuracy_category': ' '.join(map(str, pronunciation_categories)),
            'is_letter_correct_all_words': is_letter_correct_all_words.strip()
        }
        return result

# --- UTILITY FUNCTIONS FOR WORD ALIGNMENT ---
def edit_distance(seq1: str, seq2: str) -> int:
    """Calculates the Levenshtein (edit) distance between two sequences."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            cost = 0 if seq1[x - 1] == seq2[y - 1] else 1
            matrix[x, y] = min(
                matrix[x - 1, y] + 1,        # Deletion
                matrix[x - 1, y - 1] + cost, # Substitution
                matrix[x, y - 1] + 1,        # Insertion
            )
    return matrix[size_x - 1, size_y - 1]

def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.ndarray:
    """Creates a cost matrix for word alignment using edit distance with phonetic similarity."""
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)
    word_distance_matrix = np.zeros((number_of_estimated_words, number_of_real_words))

    for idx_estimated, est_word in enumerate(words_estimated):
        for idx_real, real_word in enumerate(words_real):
            # Basic edit distance
            distance = edit_distance(est_word.lower(), real_word.lower())
            max_len = max(len(est_word), len(real_word), 1)
            normalized_distance = distance / max_len
            
            # Bonus for words that start with the same letter(s)
            if est_word.lower().startswith(real_word.lower()[:2]) or real_word.lower().startswith(est_word.lower()[:2]):
                normalized_distance *= 0.8
            
            # Bonus for similar length words
            length_diff = abs(len(est_word) - len(real_word)) / max_len
            if length_diff < 0.3:  # Similar length
                normalized_distance *= 0.9
            
            word_distance_matrix[idx_estimated, idx_real] = normalized_distance

    return word_distance_matrix

def get_best_mapped_words(words_estimated: list, words_real: list) -> tuple:
    """Uses DTW to align estimated words with real words."""
    if not words_estimated or not words_real:
        return (['-'] * len(words_real), [-1] * len(words_real))

    try:
        # Clean punctuation
        words_real_clean = [word.strip(string.punctuation).lower() for word in words_real]
        words_estimated_clean = [word.strip(string.punctuation).lower() for word in words_estimated]
        
        word_distance_matrix = get_word_distance_matrix(words_estimated_clean, words_real_clean)
        
        # Use DTW for alignment
        alignment = dtw_from_distance_matrix(word_distance_matrix.T)
        
        # Get the alignment path
        path_query = alignment.path[:, 0]  # Real word indices
        path_reference = alignment.path[:, 1]  # Estimated word indices
        
        # Initialize result arrays
        mapped_words = ['-'] * len(words_real)
        mapped_words_indices = [-1] * len(words_real)
        
        # Create mapping from the DTW path, but allow multiple mappings
        used_estimated = set()
        for real_idx, est_idx in zip(path_query, path_reference):
            if 0 <= real_idx < len(words_real) and 0 <= est_idx < len(words_estimated):
                # Only map if this estimated word hasn't been used or if it's a better match
                if est_idx not in used_estimated or mapped_words[real_idx] == '-':
                    mapped_words[real_idx] = words_estimated[est_idx]
                    mapped_words_indices[real_idx] = est_idx
                    used_estimated.add(est_idx)

        # Post-process: Find good matches that DTW might have missed
        for real_idx, real_word in enumerate(words_real_clean):
            if mapped_words[real_idx] == '-':  # No mapping found by DTW
                best_match_idx = -1
                best_similarity = 0
                
                for est_idx, est_word in enumerate(words_estimated_clean):
                    if est_idx in used_estimated:
                        continue
                        
                    # Calculate similarity
                    distance = edit_distance(real_word, est_word)
                    max_len = max(len(real_word), len(est_word), 1)
                    similarity = 1 - (distance / max_len)
                    
                    # Lower threshold for good matches
                    if similarity > best_similarity and similarity > 0.6:
                        best_similarity = similarity
                        best_match_idx = est_idx
                
                if best_match_idx != -1:
                    mapped_words[real_idx] = words_estimated[best_match_idx]
                    mapped_words_indices[real_idx] = best_match_idx
                    used_estimated.add(best_match_idx)

        print(f"Real words: {words_real}")
        print(f"Estimated words: {words_estimated}")
        print(f"DTW alignment path (real->est): {list(zip(path_query, path_reference))}")
        print(f"Mapped words: {mapped_words}")
        
        return (mapped_words, mapped_words_indices)

    except Exception as e:
        print(f"Error in DTW alignment: {e}")
        # Enhanced fallback mapping
        mapped_words = ['-'] * len(words_real)
        mapped_words_indices = [-1] * len(words_real)
        
        used_estimated_indices = set()
        
        # First pass: find exact or very close matches
        for real_idx, real_word in enumerate(words_real_clean):
            best_match_idx = -1
            best_similarity = 0
            
            for est_idx, est_word in enumerate(words_estimated_clean):
                if est_idx in used_estimated_indices:
                    continue
                    
                distance = edit_distance(real_word, est_word)
                max_len = max(len(real_word), len(est_word), 1)
                similarity = 1 - (distance / max_len)
                
                if similarity > best_similarity and similarity > 0.6:  # Higher threshold for fallback
                    best_similarity = similarity
                    best_match_idx = est_idx
            
            if best_match_idx != -1:
                mapped_words[real_idx] = words_estimated[best_match_idx]
                mapped_words_indices[real_idx] = best_match_idx
                used_estimated_indices.add(best_match_idx)
        
        print(f"Fallback mapping: {mapped_words}")
        return (mapped_words, mapped_words_indices)

def get_which_letters_were_correct(real_word: str, transcribed_word: str) -> list:
    """Compares words letter by letter for front-end highlighting using better alignment."""
    is_letter_correct = []
    real_word_lower = real_word.lower()
    transcribed_word_lower = transcribed_word.lower() if transcribed_word != '-' else ''
    
    if transcribed_word == '-' or not transcribed_word_lower:
        # No transcription - mark all as incorrect except punctuation
        for char in real_word:
            if char in string.punctuation:
                is_letter_correct.append(1)
            else:
                is_letter_correct.append(0)
        return is_letter_correct
    
    # Use dynamic programming to find best character alignment
    real_len = len(real_word_lower)
    trans_len = len(transcribed_word_lower)
    
    # DP table for edit distance with traceback
    dp = np.zeros((real_len + 1, trans_len + 1), dtype=int)
    
    # Initialize
    for i in range(real_len + 1):
        dp[i][0] = i
    for j in range(trans_len + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, real_len + 1):
        for j in range(1, trans_len + 1):
            if real_word_lower[i-1] == transcribed_word_lower[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                   dp[i][j-1],      # insertion  
                                   dp[i-1][j-1])    # substitution
    
    # Traceback to find alignment
    i, j = real_len, trans_len
    real_aligned = []
    trans_aligned = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and real_word_lower[i-1] == transcribed_word_lower[j-1]:
            real_aligned.append(real_word_lower[i-1])
            trans_aligned.append(transcribed_word_lower[j-1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            real_aligned.append(real_word_lower[i-1])
            trans_aligned.append(transcribed_word_lower[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            real_aligned.append(real_word_lower[i-1])
            trans_aligned.append('-')
            i -= 1
        else:
            real_aligned.append('-')
            trans_aligned.append(transcribed_word_lower[j-1])
            j -= 1
    
    real_aligned.reverse()
    trans_aligned.reverse()
    
    # Now create the correctness array based on alignment
    for idx, char in enumerate(real_word):
        if char in string.punctuation:
            is_letter_correct.append(1)  # Punctuation is always correct
        elif idx < len(real_aligned):
            if real_aligned[idx] != '-' and real_aligned[idx] == trans_aligned[idx]:
                is_letter_correct.append(1)
            else:
                is_letter_correct.append(0)
        else:
            is_letter_correct.append(0)
    
    return is_letter_correct

# --- UTILITY FUNCTIONS ---
def generate_random_string(str_length: int = 20) -> str:
    """Generates a random string for unique temporary filenames."""
    letters = string.ascii_lowercase
    return ''.join((random.choice(letters) for _ in range(str_length)))

# --- Model Initialization ---
print("Initializing models...")
asr_model_en = WhisperASRModel(model_name='openai/whisper-base')
asr_model_fr = WhisperASRModel(model_name='openai/whisper-base')  # Same model for French
phonem_converter_en = EngPhonemConverter()
phonem_converter_fr = EpitranPhonemConverter(epitran.Epitran('fra-Latn'))  # French IPA converter

trainers = {
    'en': PronunciationTrainer(asr_model_en, phonem_converter_en),
    'fr': PronunciationTrainer(asr_model_fr, phonem_converter_fr),  # Added French support
}

# Create sample databases for both languages
database_folder = './databases'
if not os.path.exists(database_folder):
    os.makedirs(database_folder)

# Create CSV files if they don't exist
en_csv_path = os.path.join(database_folder, 'data_en.csv')
fr_csv_path = os.path.join(database_folder, 'data_fr.csv')

text_datasets = {
    'en': TextDataset(en_csv_path),
    'fr': TextDataset(fr_csv_path),  # Added French dataset
}
print("Initialization complete.")

# --- Flask Routes ---

@app.route('/get_sample', methods=['POST'])
def get_sample():
    """Provides a random sentence for practice in the specified language."""
    data = request.get_json()
    language = data.get('language', 'en')
    
    if language in text_datasets:
        sentence = text_datasets[language].get_random_sample()
        return jsonify({'sentence': sentence})
    return jsonify({'error': 'Language not supported'}), 400

@app.route('/score_pronunciation', methods=['POST'])
def score_pronunciation():
    """Main endpoint for scoring pronunciation in English or French."""
    try:
        data = request.get_json()
        real_text = data['text']
        base64_audio = data['audio']
        language = data.get('language', 'en')

        if language not in trainers:
            return jsonify({'error': 'Language not supported'}), 400

        # Decode audio
        file_bytes = base64.b64decode(base64_audio.split(',')[1])

        # Save to temp directory
        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_filename = os.path.join(temp_dir, f"audio_{generate_random_string(12)}.wav")
        with open(temp_filename, 'wb') as f:
            f.write(file_bytes)
            
        try:
            # Load and process audio
            signal, sr_native = sf.read(temp_filename)
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)
            signal = signal.flatten().astype(np.float32)
            
            # Pad/trim audio
            min_length = 16000
            max_length = 480000
            if signal.shape[0] < min_length:
                signal = np.pad(signal, (0, min_length - signal.shape[0]))
            elif signal.shape[0] > max_length:
                signal = signal[:max_length]
                
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        # Resample to 16kHz
        if sr_native != 16000:
            resampler = Resample(orig_freq=sr_native, new_freq=16000)
            signal = resampler(torch.tensor(signal, dtype=torch.float32)).numpy()

        # Process with trainer for the specified language
        trainer = trainers[language]
        result = trainer.process_audio_for_text(signal, real_text)

        return jsonify(result)

    except Exception as e:
        print(f"Error during pronunciation scoring: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process audio', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=False, threaded=True)