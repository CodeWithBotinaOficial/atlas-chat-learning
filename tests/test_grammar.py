import unittest
from atlas.grammar import GrammarHelper

class TestGrammarHelper(unittest.TestCase):

    def test_capitalize_first(self):
        self.assertEqual(GrammarHelper.capitalize_first("hello world"), "Hello world")
        self.assertEqual(GrammarHelper.capitalize_first(" Hello world"), " Hello world") # Leading space preserved
        self.assertEqual(GrammarHelper.capitalize_first("123 hello"), "123 hello") # First alpha char
        self.assertEqual(GrammarHelper.capitalize_first(""), "")
        self.assertEqual(GrammarHelper.capitalize_first("a"), "A")
        self.assertEqual(GrammarHelper.capitalize_first("A"), "A")
        self.assertEqual(GrammarHelper.capitalize_first(" hola mundo"), " Hola mundo")

    def test_add_punctuation(self):
        self.assertEqual(GrammarHelper.add_punctuation("Hello world"), "Hello world.")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world."), "Hello world.")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world!"), "Hello world!")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world?"), "Hello world?")
        self.assertEqual(GrammarHelper.add_punctuation(""), "")
        self.assertEqual(GrammarHelper.add_punctuation("  "), "  ") # Only spaces
        self.assertEqual(GrammarHelper.add_punctuation("Hello world  "), "Hello world.") # Trailing spaces trimmed

    def test_remove_excessive_repetition(self):
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hola hola hola mundo"), "hola hola mundo")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("vamos vamos vamos tu el el"), "vamos vamos tu el el")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("single"), "single")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("a a a a a"), "a a")
        self.assertEqual(GrammarHelper.remove_excessive_repetition(""), "")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("word word word word word", max_repeats=3), "word word word")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hello hello world world world"), "hello hello world world")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hello hello hello world world"), "hello hello world world")


    def test_filter_short_responses(self):
        fallback = "I'm not sure how to answer that properly."
        self.assertEqual(GrammarHelper.filter_short_responses("Hello", min_words=2, fallback_message=fallback), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("Hello world", min_words=2, fallback_message=fallback), "Hello world")
        self.assertEqual(GrammarHelper.filter_short_responses("Hi.", min_words=2, fallback_message=fallback), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses(""), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("  "), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("!"), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("Hello world", min_words=1, fallback_message=fallback), "Hello world")
        self.assertEqual(GrammarHelper.filter_short_responses(fallback, min_words=2, fallback_message=fallback), fallback) # Should not replace fallback with itself

    def test_check_for_gibberish(self):
        # Test cases for _check_for_gibberish
        # High stopword ratio, few content words
        self.assertTrue(GrammarHelper._check_for_gibberish("el la y que de en", stopword_ratio_threshold=0.3))
        self.assertTrue(GrammarHelper._check_for_gibberish("y y y el el", stopword_ratio_threshold=0.3))
        self.assertTrue(GrammarHelper._check_for_gibberish("el la", stopword_ratio_threshold=0.3)) # 2 stopwords, 0 content words
        self.assertTrue(GrammarHelper._check_for_gibberish("el la casa", stopword_ratio_threshold=0.3)) # 2 stopwords, 1 content word, ratio 0.66 > 0.3, content_word_count 1 < 3
        self.assertTrue(GrammarHelper._check_for_gibberish("el la y casa", stopword_ratio_threshold=0.3)) # 3 stopwords, 1 content word, ratio 0.75 > 0.3, content_word_count 1 < 3

        # Low stopword ratio, more content words
        self.assertFalse(GrammarHelper._check_for_gibberish("hola mundo como estas", stopword_ratio_threshold=0.3))
        self.assertFalse(GrammarHelper._check_for_gibberish("la casa es grande", stopword_ratio_threshold=0.3)) # 2 stopwords, 3 content words, ratio 0.4 < 0.3 is false, but content_word_count 3 is not < 3
        self.assertFalse(GrammarHelper._check_for_gibberish("la casa es grande", stopword_ratio_threshold=0.5)) # 2 stopwords, 3 content words, ratio 0.4 < 0.5 is true, but content_word_count 3 is not < 3
        self.assertFalse(GrammarHelper._check_for_gibberish("casa grande bonita", stopword_ratio_threshold=0.3)) # 0 stopwords, 3 content words
        self.assertFalse(GrammarHelper._check_for_gibberish("el la y la casa es grande", stopword_ratio_threshold=0.3)) # 4 stopwords, 3 content words, ratio 0.57 > 0.3, but content_word_count 3 is not < 3

        # Edge cases
        self.assertTrue(GrammarHelper._check_for_gibberish("", stopword_ratio_threshold=0.3))
        self.assertTrue(GrammarHelper._check_for_gibberish("   ", stopword_ratio_threshold=0.3))
        self.assertTrue(GrammarHelper._check_for_gibberish("!", stopword_ratio_threshold=0.3))
        self.assertFalse(GrammarHelper._check_for_gibberish("hello", stopword_ratio_threshold=0.3)) # Not a stopword

    def test_apply_all(self):
        # Example 1: Repetition, capitalization, punctuation
        self.assertEqual(GrammarHelper.apply_all("hola hola hola mundo mundo como estas"), "Hola hola mundo mundo como estas.")
        
        # Example 2: Messy input, likely to trigger fallback
        self.assertEqual(GrammarHelper.apply_all("tú qué hola qué bien aquí por haces tú aquí qué"), "I'm not sure how to answer that properly.")

        # Example 3: Short response, triggers fallback
        self.assertEqual(GrammarHelper.apply_all("hi"), "I'm not sure how to answer that properly.")
        self.assertEqual(GrammarHelper.apply_all("ok"), "I'm not sure how to answer that properly.")

        # Example 4: Already good
        self.assertEqual(GrammarHelper.apply_all("Hello, how are you?"), "Hello, how are you?")

        # Example 5: Empty string
        self.assertEqual(GrammarHelper.apply_all(""), "I'm still learning to form proper sentences. Could you rephrase?")
        self.assertEqual(GrammarHelper.apply_all("   "), "I'm still learning to form proper sentences. Could you rephrase?")

        # Example 6: Only punctuation
        self.assertEqual(GrammarHelper.apply_all("!!!"), "I'm still learning to form proper sentences. Could you rephrase?")

        # Example 7: Gibberish (high stopword ratio)
        self.assertEqual(GrammarHelper.apply_all("el la y de en por"), "I'm still learning to form proper sentences. Could you rephrase?")
        self.assertEqual(GrammarHelper.apply_all("el la casa y el"), "I'm not sure how to answer that properly.") # 4 stopwords, 1 content word, ratio 0.8 > 0.3, content_word_count 1 < 3

        # Example 8: Response with some stopwords but also content
        self.assertEqual(GrammarHelper.apply_all("la casa es grande y bonita"), "La casa es grande y bonita.")

        # Example 9: Check min_words after other corrections
        self.assertEqual(GrammarHelper.apply_all("hola hola"), "I'm not sure how to answer that properly.") # Becomes "Hola hola." (2 words) -> fallback because min_words=3

if __name__ == '__main__':
    unittest.main()
