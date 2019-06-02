import os
import re

import folders
from sites.bookcave import bookcave
from text import paragraph_io


def extract_section_paragraphs(fname, endings):
    with open(fname, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()

    section_index = -1
    sections, section_paragraphs = [], []
    has_toc = False
    current_line = ''

    for line in lines:
        line_lower = line.lower()
        # Ignore blank lines.
        if len(line_lower) == 0:
            continue
        # Ignore lines with no relevant content.
        if line_lower[-1] not in endings and re.search(r'[a-z0-9]', line_lower) is None:
            current_line = ''
            continue

        if has_toc:
            if section_index == -1:
                # Check for line that refers to the ToC or that matches the first section.
                if line_lower == 'table of contents' or line_lower == 'contents':
                    # The ToC refers to itself. Ignore all previous sections.
                    sections, section_paragraphs = [], []
                elif len(sections) > 0 and line == sections[0]:
                    # Found a line that matches the first section. Start collecting.
                    section_index += 1
                else:
                    # Found another section.
                    sections.append(line)
                    section_paragraphs.append([])
            else:
                # Collecting section content.
                if section_index + 1 < len(sections) and line == sections[section_index + 1]:
                    # Found a line that matches the next section.
                    section_index += 1
                    current_line = ''
                elif line_lower[-1] not in endings:
                    # The line ends with a non-standard ending. Save it.
                    current_line += line + ' '
                else:
                    # Add the line and anything saved before it.
                    section_paragraphs[section_index].append(current_line + line)
                    current_line = ''
        else:
            if section_index == -1:
                # Look for 'table of contents' or something similar.
                if line_lower == 'table of contents' or line_lower == 'contents':
                    has_toc = True
                elif line_lower == 'introduction':
                    pass
                elif line_lower == 'prologue':
                    pass
                elif line_lower == 'chapter 1' or line_lower == 'chapter one' or line_lower == 'chapter i':
                    pass
                elif line_lower == '1' or line_lower == 'one' or line_lower == 'i':
                    pass
                else:
                    pass
            else:
                pass

    return sections, section_paragraphs


def write_section_paragraphs(fnames, endings, Y=None, force=False):
    for i, fname in enumerate(fnames):
        last_sep_index = fname.rindex(os.sep)
        paragraphs_path = os.path.join(fname[:last_sep_index], folders.FNAME_TEXT_PARAGRAPHS)
        if os.path.exists(paragraphs_path):
            if force:
                os.remove(paragraphs_path)
            else:
                continue

        sections, section_paragraphs = extract_section_paragraphs(fname, endings)
        section_sizes = [len(paragraphs) for paragraphs in section_paragraphs]
        total_paragraphs = sum(section_sizes)
        min_paragraphs = min(section_sizes) if len(section_sizes) > 0 else -1
        max_paragraphs = max(section_sizes) if len(section_sizes) > 0 else -1
        avg_paragraphs = total_paragraphs/len(sections) if len(sections) is not 0 else -1
        print('{}: sections={:4d}; paragraphs={:4d}; min={:4d}; max={:4d}; avg={:5.1f}'.format(fname,
                                                                                               len(sections),
                                                                                               total_paragraphs,
                                                                                               min_paragraphs,
                                                                                               max_paragraphs,
                                                                                               avg_paragraphs))
        # TODO: Remove Y
        if Y is not None:
            print('{}; total={:2d}'.format(Y[:, i], sum(Y[:, i])))
            print()
        paragraph_io.write_formatted_section_paragraphs(sections, section_paragraphs, paragraphs_path)


def main():
    inputs, Y, _, _ = bookcave.get_data({'text'},
                                        text_source='book',
                                        text_input='filename',
                                        text_min_len=6)
    fnames = inputs['text']
    print('Files: {:d}'.format(len(fnames)))

    # Allow ellipsis over three dots and single/double quotes over primes.
    endings = {'.', '?', ')', '!', ':', '-', '"', ';', ',', "'", '…', '’', '”'}

    write_section_paragraphs(fnames, endings, Y=Y, force=True)


if __name__ == '__main__':
    main()
