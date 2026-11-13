from __future__ import annotations

import re

#vqa answer normalization and soft consensus accuracy, ported from the official vqa evaluation code
#(github.com/GT-Vision-Lab/VQA, PythonEvaluationTools/vqaEvaluation/vqaEval.py). the same normalize() is applied
#everywhere -> consensus answers during data selection, the grounding exact match and the vqav2 soft accuracy.
#one difference from the official script: it only normalizes the reference answers when the 10 of them are not all
#identical, here every answer is always normalized (the assignment asks for one normalization throughout)

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
NUMBER_WORDS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCTUATION = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)") #a period that is not part of a number
COMMA_STRIP = re.compile(r"(\d)(,)(\d)") #a thousands separator, 1,000 -> 1000


def process_punctuation(text):
    #punctuation next to a space (or a comma inside a number) is dropped, other punctuation becomes a space
    out = text
    for p in PUNCTUATION:
        if (p + " " in text or " " + p in text) or (re.search(COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def process_digit_article(text):
    #lowercase, number words -> digits, articles dropped, missing apostrophes restored ("dont" -> "don't")
    words = []
    for word in text.lower().split():
        word = NUMBER_WORDS.get(word, word)
        if word not in ARTICLES:
            words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def normalize(answer):
    return process_digit_article(process_punctuation(answer.replace("\n", " ").replace("\t", " ").strip()))


def soft_accuracy(prediction, references):
    #local vqa soft consensus accuracy against all ten human answers ->
    #  s(a) = 1/10 sum_i min(1, (1/3) * #{j != i : a == a_j})
    #i.e. each leave-one-out subset of nine answers gives full credit once three of them agree with the prediction
    pred = normalize(prediction)
    refs = [normalize(r) for r in references]
    return sum(min(1.0, sum(pred == refs[j] for j in range(len(refs)) if j != i) / 3) for i in range(len(refs))) / len(refs)
