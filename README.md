# AdvancedAIPronunciationScorer

# Pronunciation Scorer API

This project provides a Flask API for pronunciation scoring using Montreal Forced Aligner (MFA) for forced alignment and Facebook's wav2vec2 model for enhanced scoring.

## Setup Instructions

### 1. Install Conda

Download and install at https://www.anaconda.com/docs/getting-started/miniconda/install

### 2. Create and Activate Conda Environment

conda create -n aligner -c conda-forge montreal-forced-aligner

conda activate aligner

### 3. Install Dependencies

### 4. Download MFA Acoustic Model and Dictionary

Download the English ARPA acoustic model and dictionary using MFA commands (recommended):

mfa model download acoustic english_us_arpa

mfa model download dictionary english_us_arpa

This will place the models in your MFA `pretrained_models` directory inside the folder for user environment variables. Ensure this has been set up correctly.
MFA_ROOT_DIR which I had set to C:\Users\Daniel\Desktop\UCLL\AdvancedAI\phoneme\model

## Usage

Set up front end repository and navigate to  
/pronunciation

Send a POST request to `/score` with:

- `audio`: the audio file (WAV, 16kHz, mono)
- `transcript`: the corresponding transcript (uppercase, matching the audio)

Any results will appear in audio folder with unique timestamp. On CPU, this takes up to 3-4 minutes.

## Notes

- The API uses MFA for forced alignment and wav2vec2 for enhanced scoring.
- Ensure all dependencies are installed and models are downloaded before running.
- For best results, use clear recordings and transcripts that match the dictionary.
