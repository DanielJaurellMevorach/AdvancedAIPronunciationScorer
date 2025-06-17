import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def create_french_from_english_csv():
    """
    Creates a French dataset by translating sentences from the existing English CSV file.
    """
    # Read the existing English dataset
    en_csv_path = './databases/data_en.csv'
    
    if not os.path.exists(en_csv_path):
        print(f"Error: English dataset not found at {en_csv_path}")
        return None
    
    try:
        # Read the English CSV file
        df_english = pd.read_csv(en_csv_path, delimiter=';')
        # only first 100 rows
        df_english = df_english.head(100)
        print(f"Loaded English dataset with {len(df_english)} sentences from {en_csv_path}")
        
        # Extract sentences from the dataframe
        english_sentences = df_english['sentence'].tolist()
        
    except Exception as e:
        print(f"Error reading English CSV file: {e}")
        return None
    
    print("Initializing Helsinki-NLP English to French translation model...")
    
    # Initialize Helsinki-NLP English to French translation model
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading translation model: {e}")
        return None
    
    print(f"Translating {len(english_sentences)} English sentences to French...")
    
    french_sentences = []
    
    for i, sentence in enumerate(english_sentences):
        try:
            # Skip empty or NaN sentences
            if pd.isna(sentence) or not sentence.strip():
                print(f"  {i+1:3d}. Skipping empty sentence")
                continue
                
            # Tokenize the input sentence
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the translation
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            french_sentences.append(translated)
            
            print(f"  {i+1:3d}. EN: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            print(f"       FR: {translated[:50]}{'...' if len(translated) > 50 else ''}")
            
        except Exception as e:
            print(f"Error translating sentence {i+1}: {e}")
            # Keep the original sentence as fallback
            french_sentences.append(sentence)
            print(f"  {i+1:3d}. Translation failed, keeping original: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
    
    # Create databases directory if it doesn't exist
    database_folder = './'
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)
        print(f"Created directory: {database_folder}")
    
    # Create DataFrame and save to CSV
    df_french = pd.DataFrame({'sentence': french_sentences})
    fr_csv_path = os.path.join(database_folder, 'data_fr.csv')
    df_french.to_csv(fr_csv_path, sep=';', index=False, encoding='utf-8')
    
    print(f"\nFrench dataset created successfully!")
    print(f"Source file: {en_csv_path}")
    print(f"Output file: {fr_csv_path}")
    print(f"Number of sentences translated: {len(french_sentences)}")
    
    # Display the first few sentences for verification
    print("\nFirst 5 French sentences:")
    for i, sentence in enumerate(french_sentences[:5]):
        print(f"  {i+1}. {sentence}")
    
    return fr_csv_path

def verify_french_dataset():
    """
    Verifies that the French dataset was created correctly.
    """
    fr_csv_path = './databases/data_fr.csv'
    
    if not os.path.exists(fr_csv_path):
        print("French dataset not found!")
        return False
    
    try:
        df = pd.read_csv(fr_csv_path, delimiter=';')
        print(f"\nDataset verification:")
        print(f"  File: {fr_csv_path}")
        print(f"  Number of sentences: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for any missing or empty sentences
        empty_sentences = df['sentence'].isna().sum()
        if empty_sentences > 0:
            print(f"  Warning: {empty_sentences} empty sentences found!")
        else:
            print("  All sentences are valid!")
        
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False

if __name__ == "__main__":
    print("=== French Dataset Creator from English CSV ===")
    print("This script reads from databases/data_en.csv and creates data_fr.csv")
    print()
    
    try:
        # Create French dataset from English CSV
        csv_path = create_french_from_english_csv()
        
        if csv_path:
            # Verify the dataset
            print("\n" + "="*50)
            verify_french_dataset()
            
            print("\n" + "="*50)
            print("French dataset creation completed!")
            print("You can now use French in your pronunciation trainer.")
            print("Make sure to install epitran for French IPA conversion:")
            print("  pip install epitran")
        else:
            print("Failed to create French dataset.")
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Failed to create French dataset.")