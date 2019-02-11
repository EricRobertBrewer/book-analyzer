# Math.
import numpy as np

import bookcave


def main():
    _, Y, categories, levels = bookcave.get_data(
        input='filename',
        combine_ratings='max',
        categories_mode='soft',
        verbose=True)
    for category_index, category_name in enumerate(categories):
        y = Y[:, category_index]
        bincount = np.bincount(y)
        argmax = np.argmax(bincount)
        print('`{}` overall accuracy: {:.3%} (always guessing `{}`)'.format(category_name,
                                                                            bincount[argmax] / len(y),
                                                                            levels[category_index][argmax]))


if __name__ == '__main__':
    main()
