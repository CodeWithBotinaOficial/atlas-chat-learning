import re
from collections import Counter


class GrammarHelper:
    """
    Applies rule-based grammatical corrections to generated responses
    to improve readability and structure.
    """

    # A minimal set of Spanish stopwords for gibberish detection
    _SPANISH_STOPWORDS = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",  # Articles
        "y", "o", "u",  # Basic conjunctions
        "de", "a", "en", "por", "con", "para", "sin", "sobre",  # Common prepositions
        "que", "qué",  # Very common relative pronoun/conjunction/interrogative
    }

    @staticmethod
    def capitalize_first(text: str) -> str:
        """
        Ensures the first letter of the response is uppercase if it's an alphabetic character
        at the very beginning of the string.
        """
        if not text:
            return text
        if text[0].isalpha():  # Only capitalize if the very first character is a letter
            return text[0].upper() + text[1:]
        return text

    @staticmethod
    def add_punctuation(text: str) -> str:
        """Adds a period at the end if the sentence doesn't end with . ! ? and is not empty."""
        if not text.strip():
            return text
        # Check if the last character is already a sentence-ending punctuation
        if text.strip()[-1] not in ['.', '!', '?']:
            return text.strip() + '.'
        return text.strip()

    @staticmethod
    def remove_excessive_repetition(text: str, max_repeats: int = 2) -> str:
        """
        If a word appears more than `max_repeats` times consecutively,
        reduce to `max_repeats` occurrences.
        Example: "hola hola hola mundo" -> "hola hola mundo"
        """
        if not text:
            return text
        words = text.split()
        if not words:
            return text

        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 0
            j = i
            while j < len(words) and words[j] == current_word:
                count += 1
                j += 1

            # Add the word up to max_repeats times
            for _ in range(min(count, max_repeats)):
                cleaned_words.append(current_word)

            i = j  # Move to the next unique word sequence

        return ' '.join(cleaned_words)

    @staticmethod
    def filter_short_responses(text: str, min_words: int = 2,
                               fallback_message: str = "I'm not sure how to answer that properly.") -> str:
        """
        If the response has fewer than `min_words` words and is not a default message,
        replace with a fallback. Also, if the response is empty or only punctuation after stripping,
        it's considered too short.
        """
        # Remove punctuation for word count, but keep original text for comparison
        cleaned_text_for_word_count = re.sub(r'[^\w\s]', '', text).strip()
        words = cleaned_text_for_word_count.split()

        # Check if the response is just empty or only punctuation
        if not cleaned_text_for_word_count:
            return fallback_message

        if len(words) < min_words and text.strip() != fallback_message.strip():
            return fallback_message
        return text

    @staticmethod
    def _check_for_gibberish(text: str, stopword_ratio_threshold: float = 0.6) -> bool:
        """
        Checks if the response is gibberish based on content words, stopword ratio,
        and excessive overall word repetition.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return True # Empty text is gibberish

        total_words = len(words)
        stopword_count = sum(1 for word in words if word in GrammarHelper._SPANISH_STOPWORDS)
        content_word_count = total_words - stopword_count

        # Rule 1: Excessive Overall Word Repetition
        if total_words >= 3:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values())
            # If the most frequent word appears at least 3 times AND
            # it accounts for more than 60% of the total words, AND
            # the number of unique words is less than 3, then it's gibberish.
            if max_repetition >= 3 and (max_repetition / total_words) > 0.6 and len(word_counts) < 3:
                return True
            # Also, if a word appears 4 or more times regardless of ratio (too repetitive)
            if max_repetition >= 4:
                return True

        # Rule 3: Long phrase with any word repeated 3 or more times
        if total_words >= 8:
            word_counts = Counter(words)
            if max(word_counts.values()) >= 3:
                return True

        # Rule 2: High Stopword Ratio with Very Low Content (0 or 1 content word)
        if content_word_count < 2 and total_words > 0:
            stopword_ratio = stopword_count / total_words
            if stopword_ratio > stopword_ratio_threshold:
                return True # Gibberish due to high stopword ratio and low content

        # If none of the above gibberish conditions were met, then it's NOT gibberish.
        # This implicitly covers Rule 3: if content_word_count >= 2 and no other gibberish rule triggered,
        # then it's a valid response.
        return False

    @staticmethod
    def apply_all(text: str, previous_user_message: str = None) -> str:
        """
        Chains all grammatical correction methods.
        """
        original_text = text.strip()
        fallback_general = "I'm still learning to form proper sentences. Could you rephrase?"
        fallback_not_sure = "I'm not sure how to answer that properly."

        if not original_text:
            return fallback_general

        # 1. Check for gibberish on the original text first
        if GrammarHelper._check_for_gibberish(original_text):
            return fallback_general  # Use the more general fallback for gibberish

        # 2. Remove excessive consecutive repetition
        processed_text = GrammarHelper.remove_excessive_repetition(original_text)

        # 3. Capitalize first letter
        processed_text = GrammarHelper.capitalize_first(processed_text)

        # 4. Add punctuation
        processed_text = GrammarHelper.add_punctuation(processed_text)

        # 5. Filter short responses (after all other corrections)
        final_response = GrammarHelper.filter_short_responses(
            processed_text,
            min_words=2,
            fallback_message=fallback_not_sure
        )

        # Final check for empty or only punctuation after all processing
        if not re.sub(r'[^\w\s]', '', final_response).strip():
            return fallback_general

        return final_response