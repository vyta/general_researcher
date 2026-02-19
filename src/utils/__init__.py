"""
Shared utilities for all architectures.
Provides spell-checking and query normalization.
"""
from typing import List, Tuple
from spellchecker import SpellChecker


# Singleton spell checker instance
_spell_checker = None


def get_spell_checker() -> SpellChecker:
    """Get or create singleton SpellChecker instance."""
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = SpellChecker()
    return _spell_checker


def normalize_query(query: str) -> Tuple[str, List[str]]:
    """
    Normalize and spell-check a query string.
    
    Args:
        query: The raw query string from the user
        
    Returns:
        Tuple of (corrected_query, list of corrections made)
        
    Example:
        >>> normalize_query("artifical inteligence policy")
        ("artificial intelligence policy", ["artifical→artificial", "inteligence→intelligence"])
    """
    spell = get_spell_checker()
    words = query.split()
    corrections = []
    corrected_words = []
    
    for word in words:
        # Extract only alphabetic characters for spell checking
        stripped = ''.join(filter(str.isalpha, word))
        
        # Skip short words (likely abbreviations or intentional)
        if len(stripped) <= 3:
            corrected_words.append(word)
            continue
        
        suggestion = spell.correction(stripped)
        if suggestion and suggestion.lower() != stripped.lower():
            corrections.append(f"{word}→{suggestion}")
            # Preserve original punctuation/formatting
            corrected_words.append(word.replace(stripped, suggestion))
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words), corrections


def log_query_corrections(corrections: List[str], prefix: str = "ℹ️") -> None:
    """Print query corrections if any were made."""
    if corrections:
        print(f"{prefix}  Corrected query spelling: {', '.join(corrections)}")
