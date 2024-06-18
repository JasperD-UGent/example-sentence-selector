import spacy
from spacy.language import Language
from spacy.tokens import Doc
import sys


def init_custom_pipeline(es_model_spacy: str) -> spacy.Language:
    """Initialise spaCy pipeline on pretokenised input.
    :return: the pipeline in question, as an initialised spaCy Language object."""

    class WhitespaceTokeniser:
        def __init__(self, vocab):
            self.vocab = vocab

        def __call__(self, text):

            if text == "":
                words = []
                spaces = []
            else:
                words = text.split(" ")
                spaces = [True] * len(words)
                spaces[-1] = False

            return Doc(self.vocab, words=words, spaces=spaces)

    @Language.component("custom-sentenciser")
    def custom_sentenciser(doc):

        for tok in doc:
            tok.is_sent_start = False

        return doc

    nlp = spacy.load(es_model_spacy, disable=["ner", "textcat"])
    nlp.tokenizer = WhitespaceTokeniser(nlp.vocab)
    nlp.add_pipe("custom-sentenciser", before="parser")
    print(f"List names custom spaCy pipeline: {nlp.pipe_names}.")

    return nlp
