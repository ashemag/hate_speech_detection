"""
Source: https://norvig.com/spell-correct.html
"""
import re
from collections import Counter


class SpellChecker():
    def __init__(self):
        big_words = open('data/big.txt').read()
        self.WORDS = Counter(re.findall(r'\w+', big_words.lower()))

    def _P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self._candidates(word), key=self._P)

    def _candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self._known([word]) or self._known(self._edits1(word)) or self._known(self._edits2(word)) or [word])

    def _known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    @staticmethod
    def _edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))
