import nltk
import re


def normalize(line):
    # Replace double quotation marks with double prime.
    # Exclude double high-reversed-9 `‟` and double low-9 `„`.
    line = re.sub(r'[“”]', '"', line)

    # Replace single quotation marks with prime.
    # Exclude okina `ʻ`.
    line = re.sub(r'[‘’]', '\'', line)

    # Replace em-dashes, en-dashes, and non-breaking hyphens with hyphens.
    # Exclude minus sign `−`, horizontal bar `―`, figure dash `‒`,
    # horizontal line extension `⎯`, box drawings light horizontal character `─`, and modifier letter minus sign `˗`.
    line = re.sub(r'[—–‑]', '-', line)

    return line


def process_lines(tokenizer, lines, lower=True, sentences=False, endings=None, min_len=None, normal=True):
    for line in lines:
        # Check min_len before normalizing.
        if min_len is not None and len(line) < min_len:
            continue

        # Optionally normalize the text.
        if normal:
            line = normalize(line)

        # Validate the line by its last character.
        if endings is not None:
            if len(line) == 0 or line[-1] not in endings:
                continue

        # Optionally convert to lowercase.
        if lower:
            line = line.lower()

        # Tokenize the line or the sentences.
        if sentences:
            for sentence in nltk.sent_tokenize(line):
                yield tokenizer.tokenize(sentence)
        else:
            yield tokenizer.tokenize(line)
