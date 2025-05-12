#train.py
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# Download NLTK data (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet

def get_wordnet_pos(tag):
    """Map NLTK POS tag to WordNet POS tag"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def replace_synonyms(text, synonym_chance=0.7, max_candidates=5):
    """
    Replace words in the text with their synonyms, ensuring grammatical correctness.
    
    Args:
        text (str): The input text.
        synonym_chance (float): Probability of replacing a word with synonym.
        max_candidates (int): Maximum number of candidate synonyms to consider.
    
    Returns:
        str: The paraphrased text.
    """
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    paraphrased_words = []
    
    for i, (word, tag) in enumerate(pos_tags):
        # Skip short words, punctuation, and common words
        if len(word) <= 3 or not word.isalpha() or word.lower() in {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'as', 'be', 'on', 'with'}:
            paraphrased_words.append(word)
            continue
            
        # Only attempt synonym replacement with probability synonym_chance
        if random.random() > synonym_chance:
            paraphrased_words.append(word)
            continue
            
        wordnet_pos = get_wordnet_pos(tag)
        if not wordnet_pos:
            paraphrased_words.append(word)
            continue
            
        # Get all synsets for the word with the correct POS
        synsets = wordnet.synsets(word, pos=wordnet_pos)
        
        if not synsets:
            paraphrased_words.append(word)
            continue
            
        # Get unique lemmas from all synsets
        lemmas = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower() and "_" not in lemma.name():
                    lemmas.add(lemma.name().lower())
        
        # Convert to list and limit candidates
        lemmas = list(lemmas)[:max_candidates]
        
        if lemmas:
            # Choose a random synonym
            synonym = random.choice(lemmas)
            
            # Match original capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
                
            paraphrased_words.append(synonym)
        else:
            paraphrased_words.append(word)
    
    # Handle spacing around punctuation
    result = " ".join(paraphrased_words)
    for punct in [',', '.', '!', '?', ':', ';']:
        result = result.replace(f" {punct}", punct)
    
    return result

def reorder_phrases(sentence):
    """
    Attempt to reorder phrases in a sentence while maintaining meaning.
    Only applies to certain sentence structures.
    """
    # Simple implementation - for more complex reordering, you'd need a parser
    if ", " in sentence and random.random() > 0.5:
        parts = sentence.split(", ", 1)
        if len(parts) == 2 and not any(w in parts[0].lower() for w in ["if", "when", "while", "because"]):
            # Ensure the second part can stand as its own clause
            if any(parts[1].startswith(w) for w in ["and ", "but ", "however "]):
                return sentence  # Don't reorder in this case
            return parts[1] + ", " + parts[0]
    
    return sentence

def generate_paraphrases(text, num_paraphrases=3, variation_level="medium"):
    """
    Generate paraphrases for the given text with different variation levels.
    
    Args:
        text (str): The input text.
        num_paraphrases (int): Number of paraphrases to generate.
        variation_level (str): Level of variation - "low", "medium", or "high".
    
    Returns:
        list: A list of paraphrased sentences.
    """
    # Set parameters based on variation level
    if variation_level == "low":
        synonym_chance = 0.3
        reorder_chance = 0.1
    elif variation_level == "medium":
        synonym_chance = 0.5
        reorder_chance = 0.3
    else:  # high
        synonym_chance = 0.7
        reorder_chance = 0.5
    
    sentences = sent_tokenize(text)
    paraphrases = []
    
    for _ in range(num_paraphrases):
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Replace synonyms
            paraphrased = replace_synonyms(sentence, synonym_chance)
            
            # Potentially reorder phrases
            if random.random() < reorder_chance:
                paraphrased = reorder_phrases(paraphrased)
                
            paraphrased_sentences.append(paraphrased)
            
        paraphrased_text = " ".join(paraphrased_sentences)
        paraphrases.append(paraphrased_text)
        
    return paraphrases

# Example usage
if __name__ == "__main__":
    text = "Artificial intelligence (AI) is transforming the world in unprecedented ways. From healthcare to finance, AI is being used to solve complex problems and improve efficiency."
    
    print("Original text:")
    print(text)
    print("\nParaphrases:")
    
    for level in ["low", "medium", "high"]:
        print(f"\n{level.capitalize()} variation level:")
        paraphrases = generate_paraphrases(text, num_paraphrases=2, variation_level=level)
        for i, paraphrase in enumerate(paraphrases):
            print(f"{i+1}. {paraphrase}")