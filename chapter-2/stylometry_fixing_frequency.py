import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

LINES = ["-", ":", "--"]  # Line styles for graphs


def main():
    strings_by_author = dict()
    strings_by_author["doyle"] = text_to_string("hound.txt")
    strings_by_author["wells"] = text_to_string("war.txt")
    strings_by_author["unknown"] = text_to_string("lost.txt")

    print(strings_by_author["doyle"][:300])

    words_by_author = make_word_dict(strings_by_author)
    len_shortest_corpus = find_shortest_corpus(words_by_author)
    word_length_test(words_by_author, len_shortest_corpus)
    stopwords_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    vocab_test(words_by_author)
    jaccard_test(words_by_author, len_shortest_corpus)


def text_to_string(filename):
    """Reads a text file and returns a string."""
    with open(filename, encoding="utf-8") as infile:
        return infile.read()


def make_word_dict(strings_by_author):
    """
    Returns a dictionary containing lists of tokens
    in the form of words assigned to the corresponding author.
    """
    words_by_author = dict()
    for author in strings_by_author:
        tokens = nltk.word_tokenize(strings_by_author[author])
        words_by_author[author] = [token.lower() for token in tokens if token.isalpha()]
    return words_by_author


def find_shortest_corpus(words_by_author):
    """Returns the length of the shortest string"""
    word_count = []
    for author in words_by_author:
        word_count.append(len(words_by_author[author]))
        print(
            f'\nNumber of words for a key "{author}" = {len(words_by_author[author])}'
        )
        len_shortest_corpus = min(word_count)
        print(f"Length of shortest corpus = {len_shortest_corpus}\n")
        return len_shortest_corpus


def plot_fd_freq(
    fd,
    max_num=None,
    cumulative=False,
    title="Frequency plot",
    linewidth=2,
    linestyle=None,
    label=None,
):
    """Plotting frequencies instead of counts in FreqDist in NLTK"""
    tmp = fd.copy()
    norm = fd.N()
    for key in tmp.keys():
        tmp[key] = float(fd[key]) / norm * 100

    if max_num:
        tmp.plot(
            max_num,
            cumulative=cumulative,
            title=title,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
    else:
        tmp.plot(
            cumulative=cumulative,
            title=title,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )

    return


def word_length_test(words_by_author, len_shortest_corpus):
    """
    Creates a graph showing the frequency of word length for an author
    with the data limited to the length of the shortest corpus
    """
    by_author_lenght_freq_dist = dict()
    plt.figure(1)
    plt.ion()
    for i, author in enumerate(words_by_author):
        word_lenghts = [
            len(word) for word in words_by_author[author][:len_shortest_corpus]
        ]
        by_author_lenght_freq_dist[author] = nltk.FreqDist(word_lenghts)
        plot_fd_freq(
            by_author_lenght_freq_dist[author],
            max_num=15,
            linestyle=LINES[i],
            label=author,
            title="The frequency of words of different lengths",
        )

    plt.legend()
    plt.ylabel("Frequency of appearances [%]")
    plt.xlabel("Number of words")


def stopwords_test(words_by_author, len_shortest_corpus):
    """
    Creates a graph of the frequency of occurrence of non-indexed words
    with data limited to the length of the shortest corpus.
    """
    stopwords_by_author_freq_dist = dict()
    plt.figure(2)
    # Uses a collection to speed up processing
    stop_words = set(stopwords.words("english"))
    # print(f"Number of non-indexed words {len(stop_words)}")
    # print(f"Non-indexed words {stop_words}")

    for i, author in enumerate(words_by_author):
        stopwords_by_author = [
            word
            for word in words_by_author[author][:len_shortest_corpus]
            if word in stop_words
        ]
        stopwords_by_author_freq_dist[author] = nltk.FreqDist(stopwords_by_author)
        plot_fd_freq(
            stopwords_by_author_freq_dist[author],
            max_num=50,
            label=author,
            linestyle=LINES[i],
            title="The 50 most used non-indexed words",
        )

    plt.legend()
    plt.ylabel("Frequency of appearances [%]")
    plt.xlabel("Word")


def parts_of_speech_test(words_by_author, len_shortest_corpus):
    """
    Creates a chart of the part of speech used by the author
    """
    by_author_pos_freq_dist = dict()
    plt.figure(3)

    for i, author in enumerate(words_by_author):
        pos_by_author = [
            pos[1]
            for pos in nltk.pos_tag(words_by_author[author][:len_shortest_corpus])
        ]
        by_author_pos_freq_dist[author] = nltk.FreqDist(pos_by_author)
        plot_fd_freq(
            by_author_pos_freq_dist[author],
            max_num=35,
            label=author,
            linestyle=LINES[i],
            title="Parts of speech",
        )

    plt.legend()
    plt.ylabel("Frequency of appearances [%]")
    plt.xlabel("Parts of speech")
    plt.show(block=True)


def vocab_test(words_by_author):
    """
    Compares the vocabulary used by the authors using the chi-square test
    """
    chisquared_by_author = dict()
    for author in words_by_author:
        if author != "unknown":
            combined_corpus = words_by_author[author] + words_by_author["unknown"]
            author_proportion = len(words_by_author[author]) / len(combined_corpus)
            combined_freq_dist = nltk.FreqDist(combined_corpus)
            most_common_words = list(combined_freq_dist.most_common(1000))
            chisquared = 0
            for word, combined_count in most_common_words:
                observed_count_author = words_by_author[author].count(word)
                expected_count_author = combined_count * author_proportion
                chisquared += (
                    observed_count_author - expected_count_author
                ) ** 2 / expected_count_author
                chisquared_by_author[author] = chisquared
            print(f"Chi-square for key {author} = {chisquared:.1f}")
    most_likely_author = min(chisquared_by_author, key=chisquared_by_author.get)
    print(
        f"Considering the vocabulary, the most likely author is: {most_likely_author.capitalize()}"
    )


def jaccard_test(words_by_author, len_shortest_corpus):
    """
    Calculates Jaccard's similarity index between
    the disputed fragment and fragments of known authorship.
    """
    jaccard_by_author = dict()
    unique_words_unknown = set(words_by_author["unknown"][:len_shortest_corpus])
    authors = (author for author in words_by_author if author != "unknown")
    for author in authors:
        unique_words_author = set(words_by_author[author][:len_shortest_corpus])
        shared_words = unique_words_author.intersection(unique_words_unknown)
        jaccadr_sim = float(len(shared_words)) / (
            len(unique_words_author) + len(unique_words_unknown) - len(shared_words)
        )
        jaccard_by_author[author] = jaccadr_sim
        print(f"Jaccard index for the author {author.capitalize()} = {jaccadr_sim:.4f}")

    most_likely_author = max(jaccard_by_author, key=jaccard_by_author.get)
    print(
        f"Considering the similarity, the most likely author is: {most_likely_author.capitalize()}"
    )


if __name__ == "__main__":
    main()
