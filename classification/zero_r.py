# Math.
import numpy as np

import bookcave


def main():
    _, Y, categories, levels = bookcave.get_data(
        {'text'},
        text_source='preview',
        text_input='filename',
        categories_mode='soft',
        combine_ratings='max',
        verbose=True)
    for category_index, category_name in enumerate(categories):
        y = Y[:, category_index]
        bincount = np.bincount(y)
        argmax = np.argmax(bincount)
        print('`{}` overall accuracy: {:.3%} (`{}`)'.format(category_name,
                                                            bincount[argmax] / len(y),
                                                            levels[category_index][argmax]))


if __name__ == '__main__':
    main()
