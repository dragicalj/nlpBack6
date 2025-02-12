from better_profanity import profanity
from spellchecker import SpellChecker
import re
import torch
from gramformer import Gramformer
from difflib import SequenceMatcher  

class TextValidator:

    def __init__(self, next_validator=None):
        self.next_validator = next_validator

    def validate(self, text, errors):
        raise NotImplementedError("Subclasses must implement the validate method")

    def next(self, text, errors):

        if self.next_validator:
            self.next_validator.validate(text, errors)


class ProfanityValidator(TextValidator):

    def validate(self, text, errors):
        profanity.load_censor_words()
        errors["offensive"] = profanity.contains_profanity(text)  

        self.next(text, errors)  


class SpellingValidator(TextValidator):

    def __init__(self, next_validator=None):
        super().__init__(next_validator)
        self.spell = SpellChecker()

    def validate(self, text, errors):
        text_clean = re.sub(r'[^\w\s]', '', text)
        misspelled = self.spell.unknown(text_clean.split())

        if misspelled:
            corrections = {word: self.spell.correction(word) for word in misspelled if self.spell.correction(word)}
            corrected_text = " ".join([corrections.get(word, word) for word in text.split()])
            errors["spelling"] = True  
            text = corrected_text  
        else:
            errors["spelling"] = False  
        self.next(text, errors) 

class GrammarValidator(TextValidator):

    def __init__(self):
        super().__init__(None) 
        self.gf = Gramformer(models=1, use_gpu=torch.cuda.is_available())

    def validate(self, text, errors):
        suggestions = list(self.gf.correct(text))  

        if not suggestions:
            errors["grammar"] = False
            return self.next(text, errors)

        corrected_text = suggestions[0]

        min_length = min(len(text), len(corrected_text))
        truncated_text = text[:min_length]
        truncated_corrected_text = corrected_text[:min_length]

        text_diff = SequenceMatcher(None, truncated_text, truncated_corrected_text).ratio()

        if text_diff < 1.0:  
            errors["grammar"] = True
        else:
            errors["grammar"] = False

        if self.next_validator:
            self.next_validator.validate(text, errors)