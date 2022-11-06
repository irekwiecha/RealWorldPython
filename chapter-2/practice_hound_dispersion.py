import nltk

target_words = [
    "Holmes",
    "Watson",
    "Mortimer",
    "Henry",
    "Barrymore",
    "Stapleton",
    "Selden",
    "hound",
]


def main():
    text_to_analyse = text_to_string("hound.txt")
    analyse_text(text_to_analyse)


def text_to_string(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()


def analyse_text(text_to_analyse):
    tokens = nltk.word_tokenize(text_to_analyse)
    tokens = nltk.Text(tokens)
    dispersion = tokens.dispersion_plot(target_words)
    return dispersion


if __name__ == "__main__":
    main()
