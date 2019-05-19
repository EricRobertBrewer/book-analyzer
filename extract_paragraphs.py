import os

import bookcave
import preprocessing


FNAME_PARAGRAPHS = 'text_paragraphs.txt'
FNAME_PARAGRAPHS_NORMAL = 'text_paragraphs_normal.txt'


def extract_section_paragraphs(fname, endings):
    with open(fname, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()

    section_index = -1
    sections, section_paragraphs, section_paragraphs_normal = [], [], []
    has_toc = False
    current_line, current_line_normal = '', ''

    for line in lines:
        line_normal = preprocessing.normalize(line.lower())
        if len(line_normal) == 0:
            continue

        if has_toc:
            if section_index == -1:
                # Check for line that refers to the ToC or that matches the first section.
                if line_normal == 'table of contents' or line_normal == 'contents':
                    # The ToC refers to itself. Ignore all previous sections.
                    sections, section_paragraphs, section_paragraphs_normal = [], [], []
                elif len(sections) > 0 and line == sections[0]:
                    # Found a line that matches the first section. Start collecting.
                    section_index += 1
                else:
                    # Found another section.
                    sections.append(line)
                    section_paragraphs.append([])
                    section_paragraphs_normal.append([])
            else:
                # Collecting section content.
                if section_index + 1 < len(sections) and line == sections[section_index + 1]:
                    # Found a line that matches the next section.
                    section_index += 1
                    current_line = ''
                    current_line_normal = ''
                elif line_normal[-1] not in endings:
                    # The line ends with a non-standard ending. Save it.
                    current_line += line + ' '
                    current_line_normal += line_normal + ' '
                else:
                    # Add the line and anything saved before it.
                    section_paragraphs[section_index].append(current_line + line)
                    current_line = ''
                    section_paragraphs_normal[section_index].append(current_line_normal + line_normal)
                    current_line_normal = ''
        else:
            if section_index == -1:
                # Look for 'table of contents' or something similar.
                if line_normal == 'table of contents' or line_normal == 'contents':
                    has_toc = True
                elif line_normal == 'introduction':
                    pass
                elif line_normal == 'prologue':
                    pass
                elif line_normal == 'chapter 1' or line_normal == 'chapter one' or line_normal == 'chapter i':
                    pass
                elif line_normal == '1' or line_normal == 'one' or line_normal == 'i':
                    pass
                else:
                    pass
            else:
                pass

    return sections, section_paragraphs, section_paragraphs_normal


def write_formatted_section_paragraphs(path, sections, section_paragraphs):
    with open(path, 'w', encoding='utf-8') as fd:
        # Write the number of sections.
        fd.write(str(len(sections)) + '\n')
        # Write the section names.
        for section in sections:
            fd.write(section + '\n')
        for paragraphs in section_paragraphs:
            # Write the number of paragraphs in this section.
            fd.write(str(len(paragraphs)) + '\n')
            # Write each paragraph in this section.
            for paragraph in paragraphs:
                fd.write(paragraph + '\n')


def read_formatted_section_paragraphs(path):
    sections, section_paragraphs = [], []
    with open(path, 'r', encoding='utf-8') as fd:
        n_sections = int(fd.readline())
        for _ in range(n_sections):
            sections.append(fd.readline())
            section_paragraphs.append([])
        for i in range(n_sections):
            n_paragraphs = int(fd.readline())
            for _ in range(n_paragraphs):
                section_paragraphs[i].append(fd.readline())
    return sections, section_paragraphs


def write_section_paragraphs(fnames, endings, Y=None, force=False):
    for i, fname in enumerate(fnames):
        last_sep_index = fname.rindex(os.sep)
        paragraphs_path = os.path.join(fname[:last_sep_index], FNAME_PARAGRAPHS)
        paragraphs_normal_path = os.path.join(fname[:last_sep_index], FNAME_PARAGRAPHS_NORMAL)
        if os.path.exists(paragraphs_path) and os.path.exists(paragraphs_normal_path):
            if force:
                os.remove(paragraphs_path)
                os.remove(paragraphs_normal_path)
            else:
                continue

        sections, section_paragraphs, section_paragraphs_normal = extract_section_paragraphs(fname, endings)
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
        write_formatted_section_paragraphs(paragraphs_path, sections, section_paragraphs)
        write_formatted_section_paragraphs(paragraphs_normal_path, sections, section_paragraphs_normal)


def main():
    inputs, Y, _, _ = bookcave.get_data({'text'},
                                        text_source='book',
                                        text_input='filename',
                                        text_min_len=6)
    fnames = inputs['text']
    print('Files: {:d}'.format(len(fnames)))

    endings = {'.', '?', ')', '!', ':', '-', '"', ';', ',', '\'' '…'}

    write_section_paragraphs(fnames, endings, Y=Y, force=True)


if __name__ == '__main__':
    main()
