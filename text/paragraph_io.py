# Serialize and deserialize data structures based on paragraphs.
from __future__ import absolute_import, division, print_function, with_statement
from io import open
import os


def write_formatted_section_paragraphs(sections, section_paragraphs, path):
    with open(path, 'w', encoding='utf-8') as fd:
        # Write the number of sections.
        n_sections = len(sections)
        fd.write(str(n_sections) + '\n')
        # Write the section names.
        for section_i in range(n_sections):
            fd.write(sections[section_i] + '\n')
        for section_i in range(n_sections):
            # Write the number of paragraphs in this section.
            n_paragraphs = len(section_paragraphs[section_i])
            fd.write(str(n_paragraphs) + '\n')
            # Write each paragraph in this section.
            for paragraph_i in range(n_paragraphs):
                paragraph = section_paragraphs[section_i][paragraph_i]
                fd.write(paragraph + '\n')


def read_formatted_section_paragraphs(path):
    sections, section_paragraphs = [], []
    with open(path, 'r', encoding='utf-8') as fd:
        n_sections = int(fd.readline()[:-1])
        for _ in range(n_sections):
            sections.append(fd.readline()[:-1])
            section_paragraphs.append([])
        for section_i in range(n_sections):
            n_paragraphs = int(fd.readline()[:-1])
            for _ in range(n_paragraphs):
                paragraph = fd.readline()[:-1]
                section_paragraphs[section_i].append(paragraph)
    return sections, section_paragraphs


def write_formatted_section_paragraph_tokens(section_paragraph_tokens, path):
    with open(path, 'w', encoding='utf-8') as fd:
        n_sections = len(section_paragraph_tokens)
        fd.write(str(n_sections) + '\n')
        for section_i in range(n_sections):
            n_paragraphs = len(section_paragraph_tokens[section_i])
            fd.write(str(n_paragraphs) + '\n')
            for paragraph_i in range(n_paragraphs):
                tokens = section_paragraph_tokens[section_i][paragraph_i]
                fd.write(' '.join(tokens) + '\n')


def read_formatted_section_paragraph_tokens(path):
    section_paragraph_tokens = []
    with open(path, 'r', encoding='utf-8') as fd:
        n_sections = int(fd.readline()[:-1])
        for section_i in range(n_sections):
            section_paragraph_tokens.append([])
            n_paragraphs = int(fd.readline()[:-1])
            for _ in range(n_paragraphs):
                tokens = fd.readline()[:-1].split(' ')
                section_paragraph_tokens[section_i].append(tokens)
    return section_paragraph_tokens


def write_formatted_section_paragraph_labels(section_paragraph_labels, path, force=False, verbose=0):
    if not force and os.path.exists(path):
        if verbose:
            print('Skipping writing labels at `{}`. The file already exists.'.format(path))
        return

    with open(path, 'w', encoding='utf-8') as fd:
        n_sections = len(section_paragraph_labels)
        fd.write(str(n_sections) + '\n')
        for section_i in range(n_sections):
            n_paragraphs = len(section_paragraph_labels[section_i])
            fd.write(str(n_paragraphs) + '\n')
            for paragraph_i in range(n_paragraphs):
                label = section_paragraph_labels[section_i][paragraph_i]
                fd.write(str(label) + '\n')
    if verbose:
        print('Wrote labels at `{}`.'.format(path))


def read_formatted_section_paragraph_labels(path):
    section_paragraph_labels = []
    with open(path, 'r', encoding='utf-8') as fd:
        n_sections = int(fd.readline()[:-1])
        for section_i in range(n_sections):
            section_paragraph_labels.append([])
            n_paragraphs = int(fd.readline()[:-1])
            for _ in range(n_paragraphs):
                label = int(fd.readline()[:-1])
                section_paragraph_labels[section_i].append(label)
    return section_paragraph_labels
