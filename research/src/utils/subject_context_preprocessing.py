import re
import numpy
from collections import Counter

from .document_processing import find_all_occurrences_in_string, chars_occurrence_ratio


class TextTransformer:

    def _process(self, text):
        return text

    def process(self, text):
        return self._process(text)


class LineByLineTransformer(TextTransformer):

    def process(self, text):
        lines_to_process = []
        attribute_lines = []
        for line in text.split('\n'):
            if re.match(r'<[A-Z_]+>', line):
                attribute_lines.append(line)
            else:
                lines_to_process.append(line)
        processed_lines = self._process(lines_to_process)
        lines = processed_lines + attribute_lines
        return '\n'.join(lines)


class BlankLinesFilter1(TextTransformer):
    """deprecated"""

    def __init__(self, replacement='\n', top_n_percent=0.05, top_n_var_threshold=10, full_line_threshold=0.85):
        self._replacement = replacement
        self._top_n_percent = top_n_percent
        self._top_n_var_threshold = top_n_var_threshold
        self._full_line_threshold = full_line_threshold

    def _filter_lines(self, lines, max_length):
        threshold = int(max_length * self._full_line_threshold)
        skip_breakline = False
        filtered_lines = []
        for line in lines:
            if len(line) > 1:
                skip_breakline = False
            if not skip_breakline:
                filtered_lines.append(line)
            if (len(line) > threshold) or \
                    ((not re.match(' \.', filtered_lines[-1][::-1]) \
                      and (len(filtered_lines) > 1) and (len(filtered_lines[-2]) > threshold))):
                skip_breakline = True
        return self._replacement.join(filtered_lines)

    def _process(self, text):
        lines = [line for line in text.split('\n')]
        lengths = [len(line) for line in lines]
        sorted_lengths = numpy.sort(numpy.array(lengths))[::-1]
        num_top_n = max(int(len(sorted_lengths) * self._top_n_percent) + 1, 2)
        top_n = sorted_lengths[:num_top_n]
        var_of_top_n = numpy.var(top_n)
        if var_of_top_n < self._top_n_var_threshold:
            max_length = sorted_lengths[0]
            text = self._filter_lines(lines, max_length)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text


class BlankLinesFilter2(BlankLinesFilter1):
    """deprecated"""

    def _filter_lines(self, lines, max_length):
        threshold = int(max_length * self._full_line_threshold)
        skip_breakline = False
        filtered_text = ''
        filtered_lines = []
        for line in lines:
            if len(line) > 1:
                skip_breakline = False
            if not skip_breakline:
                filtered_text += line
                filtered_lines.append(line)
            if (len(filtered_lines[-1]) > threshold) or \
                    ((not filtered_lines[-1].strip().endswith('.')) \
                     and (len(filtered_lines) > 1) and (len(filtered_lines[-2]) > threshold)):
                skip_breakline = True
            else:
                filtered_text += '\n'
        return filtered_text


class BlankLinesFilter(TextTransformer):

    def __init__(self, replacement='\n', top_n_frequency=200, top_n_var_threshold=5, full_line_threshold=0.85,
                 min_max_line_length=70, min_sentence_small_char_ratio=0.6):
        self._replacement = replacement
        self._top_n_frequency = top_n_frequency
        self._top_n_var_threshold = top_n_var_threshold
        self._full_line_threshold = full_line_threshold
        self._min_max_line_length = min_max_line_length
        self._min_sentence_small_char_ratio = min_sentence_small_char_ratio

    def _filter_lines(self, lines, max_length):
        threshold = int(max_length * self._full_line_threshold)
        skip_breakline = False
        filtered_text = ''
        filtered_lines = []
        for i, line in enumerate(lines):
            if len(line) > 1:
                skip_breakline = False
            if not skip_breakline:
                filtered_text += line + ' '
                filtered_lines.append(line)
            if (len(filtered_lines[-1]) >= threshold) and \
                    (chars_occurrence_ratio(filtered_lines[-1]) > self._min_sentence_small_char_ratio):
                skip_breakline = True  # full line
            elif (not filtered_lines[-1].strip().endswith('.')) and \
                    (len(filtered_lines) > 1) and (len(filtered_lines[-2]) > threshold):
                skip_breakline = True  # unended sentence shorter than valid full line
            elif (len(lines) > i + 1) and \
                    (len(lines[i + 1]) >= threshold) and (not re.match('[A-Z]', lines[i + 1])):
                skip_breakline = True  # next line is full and does not start new sentence
            else:
                filtered_text += '\n'
        return filtered_text

    def _process(self, text):
        lines = [line for line in text.split('\n')]
        lengths = [len(line) for line in lines]
        sorted_lengths = numpy.sort(numpy.array(lengths))[::-1]
        max_length = sorted_lengths[0]
        if max_length >= self._min_max_line_length:
            num_full_length = (sorted_lengths > self._full_line_threshold * max_length).sum()
            num_top_n = max(num_full_length, 2)
            top_n = sorted_lengths[:num_top_n]
            var_of_top_n = numpy.std(top_n)
            if var_of_top_n < self._top_n_var_threshold:
                text = self._filter_lines(lines, max_length)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text


class BlankLinesFilter3(BlankLinesFilter):
    """deprecated"""

    def _process(self, text):
        lines = [line for line in text.split('\n')]
        lengths = [len(line) for line in lines]
        sorted_lengths = numpy.sort(numpy.array(lengths))[::-1]
        num_top_n = max(int(numpy.sqrt(len(text) / self._top_n_frequency)) + 1, 2)
        top_n = sorted_lengths[:num_top_n]
        var_of_top_n = numpy.std(top_n)
        if var_of_top_n < self._top_n_var_threshold:
            max_length = sorted_lengths[0]
            text = self._filter_lines(lines, max_length)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text


class IrrelevantLinesFilter(LineByLineTransformer):

    def __init__(self, keywords=['strana', 'stránka'], lower=True):
        self._keywords = keywords
        self._lower = lower

    def _process(self, lines):
        keywords = self._keywords
        if self._lower:
            keywords = [keyword.lower() for keyword in self._keywords]
        filtered_lines = []
        for line in lines:
            if any(keyword in (line.lower() if self._lower else line) for keyword in keywords):
                continue
            filtered_lines.append(line)
        return filtered_lines


class IrrelevantLinesRegexFilter(LineByLineTransformer):

    def __init__(self, patterns=[r'www', r'[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}']):
        self._patterns = patterns

    def _process(self, lines):
        filtered_lines = []
        for line in lines:
            if any(re.search(pattern, line) for pattern in self._patterns):
                continue
            filtered_lines.append(line)
        return filtered_lines


class TooShortLinesFilter(LineByLineTransformer):

    def __init__(self, too_short_line_threshold=5):
        self._too_short_line_threshold = too_short_line_threshold

    def _process(self, lines):
        filtered_lines = []
        for line in lines:
            if (len(line.strip()) > self._too_short_line_threshold) or (len(line) == 0):
                filtered_lines.append(line)
        return filtered_lines


class NumeralLinesFilter(LineByLineTransformer):

    def __init__(self, too_many_numerals_ratio_threshold=0.5):
        self._too_many_numerals_ratio_threshold = too_many_numerals_ratio_threshold

    def _process(self, lines):
        filtered_lines = []
        for line in lines:
            occurrences = find_all_occurrences_in_string('\d', line)
            ref_line = line.replace(' ', '')
            if (len(ref_line) == 0) or (len(occurrences) / len(ref_line) < self._too_many_numerals_ratio_threshold):
                filtered_lines.append(line)
        return filtered_lines


class AttributeExtractor(LineByLineTransformer):

    def __init__(self, attr_tag='<ATTRIBUTE>;<ATTRIBUTE/>'):
        self._tag_start, self._tag_end = attr_tag.split(';')

    def _complete_attribute_line(self, value):
        return self._tag_start + value + self._tag_end

    def _process_internal(self, *args):
        return []

    def _process(self, lines):
        attr_candidates = []
        for i, line in enumerate(lines):
            attr_candidates.extend(self._process_internal(line, i, lines))
        lines.extend(attr_candidates)
        return lines


class QuotedContractNameExtractor(AttributeExtractor):

    def __init__(self, name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                 false_keywords=['přílo', 'dále', 'jen'],
                 positive_keywords=['názvem'],
                 context_range=50, min_length=25):
        super().__init__(name_tag)
        self._false_keywords = false_keywords
        self._positive_keywords = positive_keywords
        self._context_range = context_range
        self._min_length = min_length

    def _process_internal(self, line, i, lines):
        name_candidates = []
        for match in re.finditer(r'"([^"]*)"', line):
            start = match.start(0)
            context = line[max(0, start - self._context_range):start]
            name = match.group(1)
            if (any(keyword in context.lower() for keyword in self._positive_keywords)) or \
                    ((len(name) >= self._min_length) and \
                     (not any(keyword in context.lower() for keyword in self._false_keywords))):
                name_candidates.append(self._complete_attribute_line(name))
        return name_candidates


class StructuredContractNameExtractor(AttributeExtractor):

    def __init__(self, name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                 false_keywords=['přílo', 'dále', 'jen', '"'],
                 positive_keywords=['název'],
                 context_range=50, min_length=25, delim=':.'):
        super().__init__(name_tag)
        self._false_keywords = false_keywords
        self._positive_keywords = positive_keywords
        self._context_range = context_range
        self._min_length = min_length
        self._delim = delim

    def _process_internal(self, line, i, lines):
        name_candidates = []
        if any(keyword.lower() in line.lower() for keyword in self._positive_keywords):
            line_parts = line.split(self._delim)
            if len(line_parts) > 1:
                name = self._delim.join(line_parts[1:]).strip()
                if (len(name) < self._min_length) and (len(lines) > i):
                    name = lines[i + 1]
                name = name.strip('. \t')
                if (len(name) > self._min_length) and \
                        (not any(keyword in name.lower() for keyword in self._false_keywords)):
                    name_candidates.append(self._complete_attribute_line(name))
        return name_candidates


class ItemEnumerationExtractor(AttributeExtractor):

    def __init__(self, item_tag='<ITEM>;<ITEM/>', enumeration_pattern=r'010(10)*'):
        super().__init__(item_tag)
        self._enumeration_pattern = enumeration_pattern

    def _get_longest_matching_part(self, lines_structure_pattern):
        selected_parts = []
        for match in re.finditer(self._enumeration_pattern, lines_structure_pattern):
            start = match.start(0)
            matched_pattern = numpy.array(list(match.group(0)))
            selected_lines = numpy.where(matched_pattern == '1')[0]
            selected_lines = numpy.add(selected_lines, start)
            selected_parts.append(selected_lines)
        if len(selected_parts) == 0:
            return []
        selected_parts_lengths = [len(p) for p in selected_parts]
        longest_selected_part = selected_parts[numpy.argmax(selected_parts_lengths)]
        return longest_selected_part

    def _process_internal(self, *args):
        return []

    def _process(self, lines):
        selected_lines = self._process_internal(lines)
        extended_lines = [self._complete_attribute_line(line) for line in selected_lines]
        lines.extend(extended_lines)
        return lines


class StructureItemEnumerationExtractor(ItemEnumerationExtractor):

    def __init__(self, enumeration_pattern=r'010(10)*', full_line_length=150, delim=':.', **kwargs):
        super().__init__(enumeration_pattern=enumeration_pattern, **kwargs)
        self._full_line_length = full_line_length
        self._delim = delim

    def _process_internal(self, lines):
        lengths = numpy.array([len(line) for line in lines])
        lines_structure = numpy.zeros(len(lengths), numpy.int32)
        lines_structure[lengths > 0] = 1
        lines_structure[lengths > self._full_line_length] = 2
        lines_structure_pattern = ''.join([str(l) for l in lines_structure])
        longest_selected_part = self._get_longest_matching_part(lines_structure_pattern)
        selected_lines = [lines[i] for i in longest_selected_part]
        if any('\t' in line for line in selected_lines):
            selected_lines = [line for line in selected_lines if '\t' in line]
        selected_lines = [line for line in selected_lines if self._delim not in line]
        if len(selected_lines) > 2:
            selected_lines = [line.strip('\t .') for line in selected_lines]
            return selected_lines
        return []


class CharItemEnumerationExtractor(ItemEnumerationExtractor):

    def __init__(self, enumeration_pattern=r'1+',
                 forbidden_chars='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž0123456789', **kwargs):
        super().__init__(enumeration_pattern=enumeration_pattern, **kwargs)
        self._forbidden_chars = forbidden_chars

    def _get_most_frequent_char(self, chars_frequencies):
        for c in chars_frequencies:
            if c[0].lower() not in self._forbidden_chars:
                return c[0]
        return None

    def _process_internal(self, lines):
        striped_lines = [l.strip() for l in lines]
        first_chars = [(l[0] if len(l) > 0 else None, i) for i, l in enumerate(striped_lines)]
        filtred_chars = list(filter(lambda c: c[0] != None, first_chars))
        most_frequent_first_chars = Counter([c[0] for c in filtred_chars]).most_common()
        most_frequent_first_char = self._get_most_frequent_char(most_frequent_first_chars)
        if not most_frequent_first_char:
            return []
        lines_structure = ['1' if c[0] == most_frequent_first_char else '0' for c in filtred_chars]
        lines_structure_pattern = ''.join(lines_structure)
        longest_selected_part = self._get_longest_matching_part(lines_structure_pattern)
        if len(longest_selected_part) > 2:
            selected_lines = [lines[filtred_chars[i][1]] for i in longest_selected_part]
            selected_lines = [l.replace(most_frequent_first_char, '').strip('\t .,;') for l in selected_lines]
            selected_lines = [line.strip('\t .') for line in selected_lines]
            return selected_lines
        return []


class AddLine(LineByLineTransformer):

    def __init__(self, line='=========='):
        self._line = line

    def _process(self, lines):
        lines.append(self._line)
        return lines


class ReplaceMarksTransformer:

    def __init__(self, marks_to_transform=['„', '“'], result_mark='"'):
        self._marks_to_transform = marks_to_transform
        self._result_mark = result_mark

    def process(self, text):
        for mark in self._marks_to_transform:
            text = text.replace(mark, self._result_mark)
        return text


class RegexReplaceTransformer:

    def __init__(self, pattern_to_transform=r'\n[ \t]*([\d]+.{0,1})+', result_pattern='\n'):
        self._pattern_to_transform = pattern_to_transform
        self._result_pattern = result_pattern

    def process(self, text):
        text = re.sub(self._pattern_to_transform, self._result_pattern, text)
        return text


class SubjectContextPreprocessor:

    def __init__(self, transformers=None):
        self._transformers = transformers \
            if transformers is not None else \
            [
                NumeralLinesFilter(too_many_numerals_ratio_threshold=0.5),
                TooShortLinesFilter(too_short_line_threshold=5),
                IrrelevantLinesFilter(keywords=['strana', 'stránka'], lower=True),
                IrrelevantLinesRegexFilter(patterns=[r'www', r'[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}']),  # email
                RegexReplaceTransformer(pattern_to_transform=r'\n[ \t]*(.{0,1}[\d]+.{0,1})+[ \t]*',  # paragraph numbers
                                        result_pattern='\n'),
                BlankLinesFilter(replacement='\n', top_n_frequency=200, top_n_var_threshold=5,
                                 full_line_threshold=0.85, min_max_line_length=70),
                ReplaceMarksTransformer(marks_to_transform='„“', result_mark='"'),
                RegexReplaceTransformer(pattern_to_transform=r'([^\n])[ ]*\n', result_pattern='\g<1>.\n'),
                ReplaceMarksTransformer(marks_to_transform=[':'], result_mark=':.'),
                ReplaceMarksTransformer(marks_to_transform=['..'], result_mark='.'),
                AddLine(line='\n'),
                QuotedContractNameExtractor(name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                                            false_keywords=['přílo', 'dále', 'jen'],
                                            positive_keywords=['názvem'],
                                            context_range=50, min_length=25),
                StructuredContractNameExtractor(name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                                                false_keywords=['přílo', 'dále', 'jen', '"'],
                                                positive_keywords=['název'], min_length=5, delim=':.'),
                StructureItemEnumerationExtractor(item_tag='<ITEM>;<ITEM/>', enumeration_pattern=r'010(10)*',
                                                  full_line_length=150, delim=':.'),
                CharItemEnumerationExtractor(item_tag='<ITEM>;<ITEM/>',
                                             enumeration_pattern=r'1+',
                                             forbidden_chars='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž0123456789'),
            ]

    def transform_text(self, text):
        for transformer in self._transformers:
            text = transformer.process(text)
        return text

    def process(self, text):
        text = self.transform_text(text)
        return text
