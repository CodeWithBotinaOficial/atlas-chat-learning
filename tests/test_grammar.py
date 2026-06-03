import unittest
from atlas.grammar import GrammarHelper

class TestGrammarHelper(unittest.TestCase):

    def test_capitalize_first(self):
        self.assertEqual(GrammarHelper.capitalize_first("hello world"), "Hello world")
        self.assertEqual(GrammarHelper.capitalize_first(" Hello world"),
                         " Hello world")  # Leading space preserved, no capitalization
        self.assertEqual(GrammarHelper.capitalize_first("123 hello"),
                         "123 hello")  # Starts with digit, no capitalization
        self.assertEqual(GrammarHelper.capitalize_first(""), "")
        self.assertEqual(GrammarHelper.capitalize_first("a"), "A")
        self.assertEqual(GrammarHelper.capitalize_first("A"), "A")
        self.assertEqual(GrammarHelper.capitalize_first(" hola mundo"),
                         " hola mundo")  # Leading space, no capitalization
        self.assertEqual(GrammarHelper.capitalize_first("!hello"), "!hello")  # Leading punctuation, no capitalization

    def test_add_punctuation(self):
        self.assertEqual(GrammarHelper.add_punctuation("Hello world"), "Hello world.")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world."), "Hello world.")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world!"), "Hello world!")
        self.assertEqual(GrammarHelper.add_punctuation("Hello world?"), "Hello world?")
        self.assertEqual(GrammarHelper.add_punctuation(""), "")
        self.assertEqual(GrammarHelper.add_punctuation("  "), "  ")  # Only spaces
        self.assertEqual(GrammarHelper.add_punctuation("Hello world  "), "Hello world.")  # Trailing spaces trimmed

    def test_remove_excessive_repetition(self):
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hola hola hola mundo"), "hola hola mundo")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("vamos vamos vamos tu el el"),
                         "vamos vamos tu el el")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("single"), "single")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("a a a a a"), "a a")
        self.assertEqual(GrammarHelper.remove_excessive_repetition(""), "")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("word word word word word", max_repeats=3),
                         "word word word")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hello hello world world world"),
                         "hello hello world world")
        self.assertEqual(GrammarHelper.remove_excessive_repetition("hello hello hello world world"),
                         "hello hello world world")

    def test_filter_short_responses(self):
        fallback = "I'm not sure how to answer that properly."
        self.assertEqual(GrammarHelper.filter_short_responses("Hello", min_words=2, fallback_message=fallback),
                         fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("Hello world", min_words=2, fallback_message=fallback),
                         "Hello world")
        self.assertEqual(GrammarHelper.filter_short_responses("Hi.", min_words=2, fallback_message=fallback), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses(""), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("  "), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("!"), fallback)
        self.assertEqual(GrammarHelper.filter_short_responses("Hello world", min_words=1, fallback_message=fallback),
                         "Hello world")
        self.assertEqual(GrammarHelper.filter_short_responses(fallback, min_words=2, fallback_message=fallback),
                         fallback)  # Should not replace fallback with itself

    def test_check_for_gibberish(self):
        # Test cases for _check_for_gibberish with updated logic and stopwords
        # Not gibberish (enough content words)
        self.assertFalse(GrammarHelper._check_for_gibberish("hola mundo como estas",
                                                            stopword_ratio_threshold=0.6))  # 3 content words
        self.assertFalse(GrammarHelper._check_for_gibberish("la casa es grande",
                                                            stopword_ratio_threshold=0.6))  # "casa", "grande" are content words (2)
        self.assertFalse(
            GrammarHelper._check_for_gibberish("casa grande bonita", stopword_ratio_threshold=0.6))  # 3 content words
        self.assertFalse(GrammarHelper._check_for_gibberish("el perro corre rápido",
                                                            stopword_ratio_threshold=0.6))  # "perro", "corre", "rápido" are content words (3)
        self.assertFalse(GrammarHelper._check_for_gibberish("un coche rojo",
                                                            stopword_ratio_threshold=0.6))  # "coche", "rojo" are content words (2)

        # Gibberish due to high stopword ratio and few content words
        self.assertTrue(GrammarHelper._check_for_gibberish("el la y de en por",
                                                           stopword_ratio_threshold=0.6))  # All stopwords, 0 content words
        self.assertTrue(GrammarHelper._check_for_gibberish("y y y el el",
                                                           stopword_ratio_threshold=0.6))  # All stopwords, 0 content words
        self.assertTrue(GrammarHelper._check_for_gibberish("el la un",
                                                           stopword_ratio_threshold=0.6))  # All stopwords, 0 content words
        self.assertTrue(GrammarHelper._check_for_gibberish("de la", stopword_ratio_threshold=0.6))  # 0 content words

        # Gibberish due to excessive repetition in short sentence
        self.assertTrue(
            GrammarHelper._check_for_gibberish("tú qué qué qué", stopword_ratio_threshold=0.6))  # "qué" repeats 3 times
        self.assertTrue(GrammarHelper._check_for_gibberish("hola hola hola hola",
                                                           stopword_ratio_threshold=0.6))  # "hola" repeats 4 times
        self.assertTrue(
            GrammarHelper._check_for_gibberish("el el el mundo", stopword_ratio_threshold=0.6))  # "el" repeats 3 times
        self.assertTrue(GrammarHelper._check_for_gibberish("vamos vamos vamos",
                                                           stopword_ratio_threshold=0.6))  # "vamos" repeats 3 times

        # Edge cases
        self.assertTrue(GrammarHelper._check_for_gibberish("", stopword_ratio_threshold=0.6))
        self.assertTrue(GrammarHelper._check_for_gibberish("   ", stopword_ratio_threshold=0.6))
        self.assertTrue(GrammarHelper._check_for_gibberish("!", stopword_ratio_threshold=0.6))
        self.assertFalse(
            GrammarHelper._check_for_gibberish("hello", stopword_ratio_threshold=0.6))  # Changed to assertFalse

    def test_apply_all(self):
        fallback_general = "I'm still learning to form proper sentences. Could you rephrase?"
        fallback_not_sure = "I'm not sure how to answer that properly."

        # Example 1: Repetition, capitalization, punctuation
        self.assertEqual(GrammarHelper.apply_all("hola hola hola mundo mundo como estas"),
                         "Hola hola mundo mundo como estas.")

        # Example 2: Messy input, now correctly triggers general fallback due to repetition
        self.assertEqual(GrammarHelper.apply_all("tú qué hola qué bien aquí por haces tú aquí qué"), fallback_general)

        # Example 3: Short response, triggers 'not sure' fallback
        self.assertEqual(GrammarHelper.apply_all("hi"), fallback_not_sure)
        self.assertEqual(GrammarHelper.apply_all("ok"), fallback_not_sure)

        # Example 4: Already good
        self.assertEqual(GrammarHelper.apply_all("Hello, how are you?"), "Hello, how are you?")

        # Example 5: Empty string
        self.assertEqual(GrammarHelper.apply_all(""), fallback_general)
        self.assertEqual(GrammarHelper.apply_all("   "), fallback_general)

        # Example 6: Only punctuation
        self.assertEqual(GrammarHelper.apply_all("!!!"), fallback_general)

        # Example 7: Gibberish (high stopword ratio)
        self.assertEqual(GrammarHelper.apply_all("el la y de en por"), fallback_general)
        self.assertEqual(GrammarHelper.apply_all("el la un"), fallback_general)

        # Example 8: Response with some stopwords but also content
        self.assertEqual(GrammarHelper.apply_all("la casa es grande y bonita"), "La casa es grande y bonita.")

        # Example 9: Check min_words after other corrections
        # "hola hola" (2 words) should not trigger fallback with min_words=2
        self.assertEqual(GrammarHelper.apply_all("hola hola"), "Hola hola.")

        # Example 10: Gibberish due to repetition, caught by _check_for_gibberish
        self.assertEqual(GrammarHelper.apply_all("hola hola hola"), fallback_general)
        self.assertEqual(GrammarHelper.apply_all("qué qué qué"), fallback_general)


if __name__ == '__main__':
    unittest.main()