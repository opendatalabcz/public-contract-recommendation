import re
import numpy
from collections import Counter

from recommender.component.feature.document import find_all_occurrences_in_string, chars_occurrence_ratio, DataProcessor


class TextTransformer:

    def _process(self, text):
        return text

    def process(self, text):
        return self._process(text)


class LineByLineTransformer(TextTransformer):

    def __init__(self, keep_attributes=False):
        self._keep_attributes = keep_attributes

    def process(self, text):
        lines_to_process = []
        attribute_lines = []
        if self._keep_attributes:
            lines_to_process = text.split('\n')
        else:
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
        non_empty_lengths = numpy.extract(sorted_lengths > 0, sorted_lengths)
        max_length = numpy.quantile(non_empty_lengths, q=0.95) if len(non_empty_lengths) > 0 else 0
        if max_length >= self._min_max_line_length:
            num_full_length = (sorted_lengths > self._full_line_threshold * max_length).sum()
            num_top_n = max(num_full_length, 2)
            top_n = sorted_lengths[:num_top_n]
            if len(top_n) > 3:
                top_n = top_n[:-1]
            if len(top_n) > 3:
                top_n = top_n[1:]
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

    def __init__(self, keywords=['strana', 'stránka'], max_line_length=75, lower=True):
        super().__init__()
        self._keywords = keywords
        self._max_line_length = max_line_length
        self._lower = lower

    def _process(self, lines):
        keywords = self._keywords
        if self._lower:
            keywords = [keyword.lower() for keyword in self._keywords]
        filtered_lines = []
        for line in lines:
            if len(line) < self._max_line_length:
                if any(keyword in (line.lower() if self._lower else line) for keyword in keywords):
                    continue
            filtered_lines.append(line)
        return filtered_lines


class IrrelevantLinesRegexFilter(LineByLineTransformer):

    def __init__(self, patterns=[r'www', r'[\w\-\.]+\s*@\s*([\w\-]+\.)+[\w\-]{2,4}',
                                 '(\+\d{2,3}){0,1}(\s{0,1}\d{3}){3}']):
        super().__init__()
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
        super().__init__()
        self._too_short_line_threshold = too_short_line_threshold

    def _process(self, lines):
        filtered_lines = []
        for line in lines:
            if (len(line.strip()) > self._too_short_line_threshold) or (len(line) == 0):
                filtered_lines.append(line)
        return filtered_lines


class NumeralLinesFilter(LineByLineTransformer):

    def __init__(self, too_many_numerals_ratio_threshold=0.5):
        super().__init__()
        self._too_many_numerals_ratio_threshold = too_many_numerals_ratio_threshold

    def _process(self, lines):
        filtered_lines = []
        for line in lines:
            occurrences = find_all_occurrences_in_string('\d', line)
            ref_line = line.replace(' ', '')
            if (len(ref_line) == 0) or (len(occurrences) / len(ref_line) < self._too_many_numerals_ratio_threshold):
                filtered_lines.append(line)
        return filtered_lines


class TooLongLinesTransformer(LineByLineTransformer):

    def __init__(self, forbidden_delimiters='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž0123456789',
                 special_delimiters={'-': (r'[\s,\.](-)[\s]+[^(Kč)]', 1)},
                 too_long_line_treshold=200):
        super().__init__()
        self._forbidden_delimiters = forbidden_delimiters
        self._delimiters = {delim: (re.compile(special_delimiters[delim][0]), special_delimiters[delim][1])
                            for delim in special_delimiters}
        self._too_long_line_treshold = too_long_line_treshold

    def _get_most_frequent_char(self, chars_frequencies):
        for c in chars_frequencies:
            if c[0].lower() not in self._forbidden_delimiters:
                return c[0]
        return None

    def _get_first_chars(self, lines):
        first_chars = [(l[0] if len(l) > 0 else None, i) for i, l in enumerate(lines)]
        filtered_chars = list(filter(lambda c: c[0] is not None, first_chars))
        return filtered_chars

    def _get_most_frequent_first_char(self, lines):
        first_chars = self._get_first_chars(lines)
        most_frequent_first_chars = Counter([c[0] for c in first_chars]).most_common()
        most_frequent_first_char = self._get_most_frequent_char(most_frequent_first_chars)
        return most_frequent_first_char

    def _extend_delimiters(self, additional_delimiters):
        delimiters = []
        for delim in additional_delimiters:
            if not delim:
                continue
            if delim in self._delimiters:
                delimiters.append(self._delimiters[delim])
                continue
            escaped_delimiter = re.escape(delim)
            pattern = re.compile(r'\s+' + escaped_delimiter)
            delimiters.append((pattern, 0))
        return delimiters

    def _reformat_breaklines(self, lines, delimiter):
        delimiters = self._extend_delimiters([delimiter])
        for pattern, group_num in delimiters:
            reformated_lines = []
            for line in lines:
                if len(line) > self._too_long_line_treshold:
                    occs = [m.start(group_num) for m in pattern.finditer(line)]
                    for m in re.finditer(r'"([^"]*)"', line):
                        occs = [occ for occ in occs if not (m.start(0) < occ < m.end(0))]
                    parts = [line[i:j] for i, j in zip([None] + occs, occs + [None])]
                    reformated_lines.extend(parts)
                else:
                    reformated_lines.append(line)
            lines = reformated_lines
        return lines

    def _process(self, lines):
        striped_lines = [l.strip() for l in lines]
        most_frequent_first_char = self._get_most_frequent_first_char(striped_lines)
        transformed_lines = self._reformat_breaklines(lines, most_frequent_first_char)
        return transformed_lines


class AttributeExtractor(LineByLineTransformer):

    def __init__(self, attr_tag='<ATTRIBUTE>;<ATTRIBUTE/>', keep_text=True, keep_attributes=False):
        super().__init__(keep_attributes=keep_attributes)
        self._tag_start, self._tag_end = attr_tag.split(';')
        self._keep_text = keep_text

    def _complete_attribute_line(self, value):
        return self._tag_start + value + self._tag_end

    def _process_internal(self, line, i, lines):
        if re.match(r'<[A-Z_]+>.*<[A-Z_]+/>', line):
            return [line]
        return []

    def _process(self, lines):
        attr_candidates = []
        for i, line in enumerate(lines):
            attr_candidates.extend(self._process_internal(line, i, lines))
        if not self._keep_text:
            return attr_candidates
        lines.extend(attr_candidates)
        return lines


class AttributeTagger(AttributeExtractor):

    def __init__(self, attr_tag='<ATTRIBUTE>;<ATTRIBUTE/>', keep_text=False, **kwargs):
        super().__init__(attr_tag, keep_text, **kwargs)

    def _process_internal(self, line, i, lines):
        return [self._complete_attribute_line(line)]


class AttributeTagCleaner(LineByLineTransformer):

    def __init__(self, attr_pattern=r'<[A-Z_]+>(.*)<[A-Z_]+/>', keep_attributes=True):
        super().__init__(keep_attributes)
        self._attr_pattern = re.compile(attr_pattern)

    def _process(self, lines):
        cleaned_lines = []
        for line in lines:
            match = self._attr_pattern.search(line)
            if match:
                cleaned_lines.append(match.group(1))
        return cleaned_lines


class QuotedContractNameExtractor(AttributeExtractor):

    def __init__(self, name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                 false_keywords=['přílo', 'dále', 'jen'],
                 positive_keywords=['názvem'],
                 context_range=50, min_length=25, **kwargs):
        super().__init__(name_tag, **kwargs)
        self._false_keywords = false_keywords
        self._positive_keywords = positive_keywords
        self._context_range = context_range
        self._min_length = min_length
        self._pattern = re.compile(r'"([^"]*)"')

    def _process_internal(self, line, i, lines):
        name_candidates = []
        for match in self._pattern.finditer(line):
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
                 positive_keywords=['název', 'stavb', 'projekt', 'předmět'],
                 context_range=30, min_length=25, delims=[':.'], **kwargs):
        super().__init__(name_tag, **kwargs)
        self._false_keywords = false_keywords
        self._positive_keywords = positive_keywords
        self._context_range = context_range
        self._min_length = min_length
        self._patterns = [re.compile(delim) for delim in delims]

    def _process_internal(self, line, i, lines):
        name_candidates = []
        for pattern in self._patterns:
            for match in pattern.finditer(line):
                start = match.start(0)
                end = match.end(0)
                context = line[max(0, start - self._context_range):start]
                if any(keyword.lower() in context.lower() for keyword in self._positive_keywords):
                    name = line[end:].strip()
                    if (len(name) < self._min_length) and (len(lines) > i + 1):
                        name = lines[i + 1]
                    name = name.strip('. \t')
                    if (len(name) > self._min_length) and \
                            (not any(keyword in name.lower() for keyword in self._false_keywords)):
                        name_candidates.append(self._complete_attribute_line(name))
        return name_candidates


class ItemExtractor(AttributeExtractor):

    def __init__(self, item_tag='<ITEM>;<ITEM/>', **kwargs):
        super().__init__(item_tag, **kwargs)

    def _process_internal(self, *args):
        return []

    def _process(self, lines):
        selected_lines = self._process_internal(lines)
        extended_lines = [self._complete_attribute_line(line) for line in selected_lines if len(line) != 0]
        lines.extend(extended_lines)
        return lines


class ItemColonExtractor(ItemExtractor):

    def __init__(self, patterns=[r'(zboží|položk).{,10}:'], **kwargs):
        super().__init__(**kwargs)
        self._patterns = [re.compile(p) for p in patterns]

    def _process_internal(self, lines):
        selected_lines = []
        for line in lines:
            for pat in self._patterns:
                match = pat.search(line)
                if match:
                    start = match.end(0)
                    item = line[start:]
                    if len(item) > 5:
                        selected_lines.append(item)
        selected_lines = [line.strip('\t .') for line in selected_lines]
        return selected_lines


class CPVCodeExtractor(AttributeExtractor):

    def __init__(self, cpv_tag='<CPV>;<CPV/>', pattern=r'(^|[^\d])([\d]{8}-\d)($|[^\d])', **kwargs):
        super().__init__(cpv_tag, **kwargs)
        self._pattern = re.compile(pattern)

    def _process_internal(self, line, i, lines):
        codes = []
        for match in self._pattern.finditer(line):
            code = match.group(2)
            codes.append(code)
        return codes


class ItemEnumerationExtractor(ItemExtractor):

    def __init__(self, enumeration_pattern=r'010(10)*', full_line_length=100, delim=':.', **kwargs):
        super().__init__(**kwargs)
        self._enumeration_pattern = enumeration_pattern
        self._full_line_length = full_line_length
        self._delim = delim

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

    def _extend_selected_lines(self, lines, selected_indexes):
        if len(selected_indexes) == 0:
            return []
        enum_start_pos = selected_indexes[0]
        lines_to_check = lines[max(0, enum_start_pos - 5):enum_start_pos][::-1]
        selected_lines = []
        for line in lines_to_check:
            if self._delim in line:
                selected_lines.append(line.split(self._delim)[-1])
                return selected_lines[::-1]
            if len(line) > self._full_line_length:
                return []
            if len(line) > 5:
                selected_lines.append(line)
        return []


class StructureItemEnumerationExtractor(ItemEnumerationExtractor):

    def __init__(self, enumeration_pattern=r'010(10)*', delim=':.',
                 **kwargs):
        super().__init__(enumeration_pattern=enumeration_pattern, delim=delim, **kwargs)
        self._upper_case_chars = 'AÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽ '

    def _upper_case_lines(self, lines):
        upper_case_lines = []
        for line in lines:
            rat = chars_occurrence_ratio(line, self._upper_case_chars)
            upper_case_lines.append(rat > 0.95)
        return upper_case_lines

    def _smooth_structure(self, structure):
        for i, f in enumerate(structure):
            if f == 2:
                if sum(structure[max(0,i-5):min(i+5, len(structure))]) < 10:
                    structure[i] = 1
        return structure

    def _process_internal(self, lines):
        lengths = numpy.array([len(line) for line in lines])
        lines_structure = numpy.zeros(len(lengths), numpy.int32)
        lines_structure[lengths > 5] = 1
        lines_structure[lengths > self._full_line_length] = 2
        lines_structure[lengths > self._full_line_length * 1.5] = 5
        lines_structure[self._upper_case_lines(lines)] = 5
        lines_structure = self._smooth_structure(lines_structure)
        lines_structure_pattern = ''.join([str(l) for l in lines_structure])
        longest_selected_part = self._get_longest_matching_part(lines_structure_pattern)
        selected_lines = self._extend_selected_lines(lines, longest_selected_part)
        selected_lines.extend([lines[i] for i in longest_selected_part])
        if any('\t' in line for line in selected_lines):  # filtering by tabular format
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
        filtered_chars = list(filter(lambda c: c[0] is not None, first_chars))
        most_frequent_first_chars = Counter([c[0] for c in filtered_chars]).most_common()
        most_frequent_first_char = self._get_most_frequent_char(most_frequent_first_chars)
        if not most_frequent_first_char:
            return []
        lines_structure = ['1' if c[0] == most_frequent_first_char else '0' for c in filtered_chars]
        lines_structure_pattern = ''.join(lines_structure)
        longest_selected_part = self._get_longest_matching_part(lines_structure_pattern)
        if len(longest_selected_part) > 2:
            selected_lines_indexes = [filtered_chars[i][1] for i in longest_selected_part]
            selected_lines = self._extend_selected_lines(lines, selected_lines_indexes)
            selected_lines.extend([lines[i] for i in selected_lines_indexes])
            selected_lines = [l.replace(most_frequent_first_char, '').strip('\t .,;') for l in selected_lines]
            selected_lines = [line.strip('\t .') for line in selected_lines]
            return selected_lines
        return []


class CharTupleItemEnumerationExtractor(ItemEnumerationExtractor):

    def __init__(self, enumeration_pattern=r'1+', min_char_occurrences=4,
                 forbidden_tuples=['Vy', 'Př', 'Za', 'Pr', 'Ob', 'Po', '| '], **kwargs):
        super().__init__(enumeration_pattern=enumeration_pattern, **kwargs)
        self._forbidden_tuples = forbidden_tuples
        self._min_char_occurrences = min_char_occurrences

    def _get_most_frequent_chars(self, chars_frequencies):
        most_frequent_chars = []
        for c in chars_frequencies:
            if c[0] not in self._forbidden_tuples and c[1] >= self._min_char_occurrences:
                most_frequent_chars.append(c[0])
        return most_frequent_chars

    def _get_selected_lines(self, most_frequent_first_char, lines, filtered_chars):
        lines_structure = ['1' if c[0] == most_frequent_first_char else '0' for c in filtered_chars]
        lines_structure_pattern = ''.join(lines_structure)
        longest_selected_part = self._get_longest_matching_part(lines_structure_pattern)
        if len(longest_selected_part) > 2:
            selected_lines_indexes = [filtered_chars[i][1] for i in longest_selected_part]
            selected_lines = self._extend_selected_lines(lines, selected_lines_indexes)
            selected_lines.extend([lines[i] for i in selected_lines_indexes])
            selected_lines = [l.replace(most_frequent_first_char, '').strip('\t .,;') for l in selected_lines]
            selected_lines = [line.strip('\t .') for line in selected_lines]
            return selected_lines
        return []

    def _process_internal(self, lines):
        striped_lines = [l.strip() for l in lines]
        first_chars = [(l[0:2] if len(l) > 1 else None, i) for i, l in enumerate(striped_lines)]
        filtered_chars = list(filter(lambda c: c[0] is not None, first_chars))
        most_frequent_first_chars = Counter([c[0] for c in filtered_chars]).most_common()
        most_frequent_first_chars = self._get_most_frequent_chars(most_frequent_first_chars)
        selected_lines = []
        for most_frequent_first_char in most_frequent_first_chars:
            selected_lines.extend(self._get_selected_lines(most_frequent_first_char, lines, filtered_chars))
        return selected_lines


class HeaderItemEnumerationExtractor(ItemEnumerationExtractor):

    def __init__(self, header_pattern='[Pp]ředmět.{0,20}je.{0,5}$', **kwargs):
        super().__init__(**kwargs)
        self._header_pattern = header_pattern

    def _process_internal(self, lines):
        selected_lines = []
        select_line = False
        for line in lines:
            if len(line) < 5:
                select_line = False
            if select_line:
                selected_lines.append(line)
            if re.search(self._header_pattern, line):
                select_line = True
        selected_lines = [line.strip('\t .') for line in selected_lines]
        return selected_lines


class AddLine(LineByLineTransformer):

    def __init__(self, line='=========='):
        super().__init__()
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


class SubjectContextPreprocessor(DataProcessor):

    def __init__(self, transformers=None, **kwargs):
        super().__init__(**kwargs)
        self._transformers = transformers \
            if transformers is not None else \
            [
                NumeralLinesFilter(too_many_numerals_ratio_threshold=0.5),
                TooShortLinesFilter(too_short_line_threshold=5),
                IrrelevantLinesFilter(keywords=['strana', 'stránka', 'e-mail'], max_line_length=75, lower=True),
                IrrelevantLinesFilter(keywords=['Tel:', 'Fax:', 'IČ:', 'IČO:', 'DIČ:'], max_line_length=75,
                                      lower=False),
                IrrelevantLinesRegexFilter(patterns=[r'www', r'[\w\-\.]+\s*@\s*([\w\-]+\.)+[\w\-]{2,4}']),  # email
                IrrelevantLinesRegexFilter(patterns=[r'(\+\d{2,3}){0,1}(\s{0,1}\d{3}){3}']),  # phone
                RegexReplaceTransformer(pattern_to_transform=r',([\s]+[A-Z][a-z ])', result_pattern='.\g<1>'),  # . vs , correction
                RegexReplaceTransformer(pattern_to_transform=r'\n[ \t]*([^\d]{0,1}[\d]{1,2}[^\d])+[ \t]*',  # paragraph numbers
                                        result_pattern='\n'),
                BlankLinesFilter(replacement='\n', top_n_frequency=200, top_n_var_threshold=5,
                                 full_line_threshold=0.85, min_max_line_length=0),
                ReplaceMarksTransformer(marks_to_transform='„“', result_mark='"'),
                TooLongLinesTransformer(forbidden_delimiters='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž0123456789',
                                        special_delimiters={'-': (r'[\s,\.](-)[\s]+[^(Kč)]', 1)},
                                        too_long_line_treshold=200),
                RegexReplaceTransformer(pattern_to_transform=r'\([^\n()]*\)', result_pattern=''),  # bracket erasing
                RegexReplaceTransformer(pattern_to_transform=r'([^\n ])[ ]*\n', result_pattern='\g<1>.\n'),  # . filling
                RegexReplaceTransformer(pattern_to_transform=r'(([Nn]ázev|[Pp]opis)[^\n,.:"()]{5,})(\s[A-Z][^\n:]{10})',
                                        result_pattern='\g<1>:\g<3>'),  # : filling
                ReplaceMarksTransformer(marks_to_transform=[':'], result_mark=':.'),
                ReplaceMarksTransformer(marks_to_transform=[';.', ',.'], result_mark='.'),
                ReplaceMarksTransformer(marks_to_transform=['..'], result_mark='.'),
                RegexReplaceTransformer(pattern_to_transform=r'[ ]*\.', result_pattern='.'),
                AddLine(line='\n'),
                StructureItemEnumerationExtractor(item_tag='<ITEM>;<ITEM/>', enumeration_pattern=r'010(10)*',
                                                  full_line_length=100, delim=':.'),
                CharTupleItemEnumerationExtractor(item_tag='<ITEM>;<ITEM/>',
                                                  enumeration_pattern=r'1+', min_char_occurrences=4,
                                                  forbidden_tuples=['Vy', 'Př', 'Za', 'Pr', 'Ob', 'Po']),
                ItemColonExtractor(item_tag='<ITEM>;<ITEM/>', patterns=[r'(zboží|položk).{,10}:']),
                HeaderItemEnumerationExtractor(item_tag='<ITEM>;<ITEM/>', header_pattern='[Pp]ředmět.{0,20}je.{0,5}$'),
                QuotedContractNameExtractor(name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                                            false_keywords=['přílo', 'dále', 'jen'],
                                            positive_keywords=['názvem'],
                                            context_range=50, min_length=25),
                StructuredContractNameExtractor(name_tag='<CONTRACT_NAME>;<CONTRACT_NAME/>',
                                                false_keywords=['přílo', 'dále', 'jen', '"'],
                                                positive_keywords=['název', 'stavb', 'projekt', 'předmět'],
                                                min_length=25, delims=[':.']),
            ]

    def transform_text(self, text):
        for transformer in self._transformers:
            text = transformer.process(text)
        return text

    def _process_inner(self, text):
        return self.transform_text(text)