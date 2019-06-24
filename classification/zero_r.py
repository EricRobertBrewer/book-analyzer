# Math.
import numpy as np

from sites.bookcave import bookcave


def main():
    min_len, max_len = 250, 7500
    _, Y, categories, category_levels =\
        bookcave.get_data({'paragraphs'},
                          min_len=min_len,
                          max_len=max_len)
    print(Y.shape)

    for category_index, category in enumerate(categories):
        y = Y[category_index]
        bincount = np.bincount(y)
        argmax = np.argmax(bincount)
        levels = category_levels[category_index]
        print('`{}` overall accuracy: {:.4f} (`{}`)'.format(category,
                                                            bincount[argmax] / len(y),
                                                            levels[argmax]))


if __name__ == '__main__':
    main()
