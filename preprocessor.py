import re


class Preprocessor():
    def __init__(self):
        self.text = None
        self.tokens = None

    def clean(self, text):
        #  remove mentions
        text = re.sub(r'@\w+', '', text)

        #  remove URLs
        text = re.sub(r'http.?://[^\s]+[\s]?', '', text)

        #  remove symbols and digits
        text = re.sub('[^a-zA-Z\s]', '', text)

        #  remove extra white spaces
        # text = re.sub("\s+", '', text)
        text = text.lstrip()
        text = text.rstrip()

        text = text.lower()

        #  spell check
        # words = text.split(' ')
        # for i, word in enumerate(words):
        #     words[i] = SpellChecker().correction(word)
        # text = ' '.join(words)
        self.text = text

    def tokenize(self):
        if self.text is None:
            raise ValueError("Call clean first.")
        self.tokens = self.text.split(' ')

    def filter_stopwords(self):
        stop_words = set(line.strip() for line in open('data/NLTK_stopwords.txt'))
        output = []
        for i, word in enumerate(self.tokens):
            if word in stop_words:
                continue
            output.append(word)
        self.tokens = output
