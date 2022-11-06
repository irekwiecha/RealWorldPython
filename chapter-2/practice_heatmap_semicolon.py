import math
from string import punctuation

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


def main():
    strings_by_author = dict()
    strings_by_author["doyle"] = text_to_string("hound.txt")
    strings_by_author["wells"] = text_to_string("war.txt")
    strings_by_author["unknown"] = text_to_string("lost.txt")
    punct_by_author = make_punct_dict(strings_by_author)
    plt.ion()
    for author in punct_by_author:
        heat = convert_punct_to_number(punct_by_author, author)
        arr = np.array((heat[:6561]))
        arr_reshaped = arr.reshape(int(math.sqrt(len(arr))), int(math.sqrt(len(arr))))
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(
            arr_reshaped, cmap=ListedColormap(["blue", "yellow"]), square=True, ax=ax
        )
        ax.set_title(f"Semicolon map for key {author.capitalize()}")
    plt.show(block=True)


def text_to_string(filename):
    with open(filename, encoding="utf-8") as infile:
        return infile.read()


def make_punct_dict(strings_by_author):
    punct_by_author = dict()
    for author in strings_by_author:
        tokens = nltk.word_tokenize(strings_by_author[author])
        punct_by_author[author] = [ch for ch in tokens if ch in punctuation]
        print(
            f"Number of punctuation marks for the key {author} = {len(punct_by_author[author])}"
        )
    return punct_by_author


def convert_punct_to_number(punct_by_author, author):
    heat = []
    for ch in punct_by_author[author]:
        if ch == ";":
            val = 1
        else:
            val = 2
        heat.append(val)
    return heat


if __name__ == "__main__":
    main()
