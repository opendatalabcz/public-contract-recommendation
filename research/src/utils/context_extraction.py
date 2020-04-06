import glob
import re
import pandas
import numpy

from .document_processing import find_all_occurrences_in_string, get_current_line, flatten_column, \
    chars_occurrence_ratio

DEF_KEYWORDS = {
    'Předmět smlouvy': 10,
    'Předmět díla': 10,
    'Předmět plnění': 10,
    'Předmět veřejné zakázky': 10,
    'Vymezení předmětu': 10,
    'Vymezení plnění': 10,
    'Popis předmětu': 10,
    'Název veřejné zakázky': 3,
    'Veřejná zakázka': 1,
    'Veřejné zakázce': 1,
    'Předmět': 1,
    'Popis': 1
}

class SubjectContextExtractor:

    def __init__(self, keywords=DEF_KEYWORDS, subj_range=2040):
        self._keywords = keywords
        self._subj_range = subj_range

    def get_all_occurrences(self, text):
        occurrences = []
        for keyword in self._keywords:
            occ = find_all_occurrences_in_string(keyword, text)
            for o in occ:
                rat = self._keywords[keyword]
                koef = 1
                matched = keyword.lower()
                current_line = get_current_line(text, o)
                # Whole line
                if current_line.lower() == keyword.lower():
                    koef += 2
                # Exact pattern match
                if text[o:min(o + len(keyword), len(text))] == keyword:
                    koef += 1.5
                    matched = keyword
                # Upper case pattern match
                if text[o:min(o + len(keyword), len(text))] == keyword.upper():
                    koef += 1.5
                    matched = keyword.upper()
                # Nearly linebreak after the pattern (chapter title)
                if '\n' in text[o:min(o + len(keyword) * 3, len(text))]:
                    koef += 2
                # Newline followed by a number preceding the pattern (chapter numbering)
                if re.search(r"\n[ ]*[0-9]", text[max(o - 20, 0):o]):
                    koef += 2
                # Nearly verb ' je ' after the pattern (subject sentence matching)
                if ' je ' in text[o:min(o + len(keyword) * 2, len(text))]:
                    koef += 2
                # Word 'článek' preceding the pattern (chapter header)
                if 'článek' in text[max(o - 20, 0):o].lower():
                    koef += 2
                # Chars 'I' preceding the pattern (chapter numbering)
                if text[max(o - 20, 0):o].count('I') > 1:
                    koef += 2
                # Nearly noun 'Zbozi' after the pattern ()
                if 'Zboží' in text[o: min(o + 150, len(text))]:
                    koef *= 0.5
                # Simple sentences following
                koef += chars_occurrence_ratio(text[min(o + 50, len(text)): min(o + 100, len(text))])

                rat *= koef
                occurrences.append({'keyword': matched, 'rat': rat, 'occ': o})
        return occurrences

    def get_subject_context_old(self, text):
        occurrences = self.get_all_occurrences(text)
        df = pandas.DataFrame(occurrences).T
        df = df.sort_values('rat')
        df = flatten_column(df, 'occ')
        if len(df.index) > 0:
            occ = df.iloc[0]
        else:
            return None
        start = occ['occ_flat'] + len(occ.name)
        end = min(start + self._subj_range, len(text))
        subj_context = text[start:end]
        return subj_context

    def get_subject_context_starts(self, text):
        bin_size = int(self._subj_range / 3)
        bins = int(len(text) / bin_size) + 1
        text = text.ljust(bins * bin_size)

        occurrences = self.get_all_occurrences(text)
        df = pandas.DataFrame(occurrences, columns=['keyword', 'rat', 'occ'])

        hist = numpy.histogram(df['occ'], bins, weights=df['rat'], range=(0, len(text)))

        score = numpy.convolve(hist[0], numpy.array([1, 1, 2, -2, -1, -1]))[2:-3]

        max_score = score.max()
        threshold = max_score * 2 / 3
        top_bins = numpy.argwhere(score > threshold).flatten()

        starts = [subj_bin * bin_size for subj_bin in top_bins]
        return starts

    def get_subject_context_end(self, text, start):
        return min(start + self._subj_range, len(text))

    def get_subject_context_ends(self, text, starts):
        return [self.get_subject_context_end(text, start) for start in starts]

    def get_subject_contexts(self, text):
        starts = self.get_subject_context_starts(text)
        ends = self.get_subject_context_ends(text, starts)
        subj_contexts = [text[start:end] for start, end in zip(starts, ends)]
        return subj_contexts

    def process(self, text):
        return self.get_subject_contexts(text)

    def show_occurrence_distribution(self, text):
        bin_size = int(self._subj_range / 3)
        bins = int(len(text) / bin_size) + 1
        text = text.ljust(bins * bin_size)

        occurrences = self.get_all_occurrences(text)
        df = pandas.DataFrame(occurrences, columns=['keyword', 'rat', 'occ'])

        hist = numpy.histogram(df['occ'], bins, weights=df['rat'], range=(0, len(text)))
        score = numpy.convolve(hist[0], numpy.array([1, 1, 2, -2, -1, -1]))[2:-3]

        sorted_bins = numpy.argsort(score)[::-1]
        top_bins = sorted_bins[:min(5, len(sorted_bins))]
        starts = [subj_bin * bin_size for subj_bin in top_bins]

        ticks = [str(s) + ' (' + str(b) + ')' for b, s in zip(top_bins, starts)]
        ax = pandas.DataFrame(hist[0]).plot.bar(figsize=(15, 5), xticks=top_bins)
        ax.set_xticklabels(ticks, rotation=45)
        ax = pandas.DataFrame(score).plot.bar(figsize=(15, 7), xticks=top_bins)
        ax.set_xticklabels(ticks, rotation=45)


class SubjectContextEndDetector:

    def __init__(self, text, subj_range=2040, multip=3):
        self._text = text
        self._subj_range = subj_range
        self._multip = multip
        self._end_position = None

    def detect(self, start_position, end_position=None):
        self._end_position = min(start_position + self._subj_range * self._multip, len(self._text))
        return self._end_position


class SubjectContextEndNumberingDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # klasicke cislovani \n<num>. word
        subj_prefix = self._text[max(start_position - 50, 0):start_position]
        numeral_pattern = '\n[ \t]*[\d]+[^/]'
        start_occurrences = find_all_occurrences_in_string(numeral_pattern, subj_prefix, lower=False)
        if len(start_occurrences) > 0:
            # otocim text abych hledal prechazejici cislo, najdu cislo nasledujici newline a vratim ho
            article_num = int(
                re.search('[\d]+', re.search('[^/][\d]+[ \t]*\n', subj_prefix[::-1]).group(0)).group(0)[::-1])
            end_occurrences = find_all_occurrences_in_string(numeral_pattern, self._text[start_position:end_position],
                                                             lower=False)
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = start_position + occ
                    m = re.search('[\d]+', self._text[occ:occ + 10])
                    if m is None:
                        continue
                    num = int(m.group(0))
                    if (num > article_num) and (num < article_num + 3):
                        current_line = get_current_line(self._text, occ + 1)
                        if len(current_line) < 50:
                            num_numbers_in_line = len(find_all_occurrences_in_string('\d', current_line))
                            if num_numbers_in_line <= 5:
                                self._end_position = occ
                                return self._end_position
        return None


class SubjectContextEndRomanNumberingDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # rimske cislovani
        subj_prefix = self._text[max(start_position - 50, 0):start_position]
        roman_numeral_pattern = '\s(?=[XVI])(X{0,3})(I[XV]|V?I{0,3})[\s\W]+'
        start_occurrences = find_all_occurrences_in_string(roman_numeral_pattern, subj_prefix, lower=False)
        if len(start_occurrences) > 0:
            end_occurrences = find_all_occurrences_in_string(roman_numeral_pattern,
                                                             self._text[start_position:end_position], lower=False)
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = start_position + occ
                    if len(get_current_line(self._text, occ + 1)) < 50:
                        self._end_position = occ
                        return self._end_position
        return None


class SubjectContextEndHeaderDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # clanek header
        subj_prefix = self._text[max(start_position - 50, 0):start_position]
        article_pattern = 'článek'
        start_occurrences = find_all_occurrences_in_string(article_pattern, subj_prefix)
        if len(start_occurrences) > 0:
            end_occurrences = find_all_occurrences_in_string(article_pattern, self._text[start_position:end_position])
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = start_position + occ
                    if len(get_current_line(self._text, occ)) < 50:
                        self._end_position = occ
                        return self._end_position
        return None


class SubjectContextEndCapitalsDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # velka pismena
        capitals_pattern = '[A-ZĚŠČŘŽÝÁÍÉÚŮŇŤÓĎ ]{4,}'
        start_line = get_current_line(self._text, start_position)
        start_match = re.search(capitals_pattern, start_line)
        if start_match and (len(start_match.group(0)) / len(start_line) > 0.5):
            new_start_position = start_position + len(start_line)
            end_occurrences = find_all_occurrences_in_string(capitals_pattern,
                                                             self._text[new_start_position:end_position])
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = new_start_position + occ
                    end_line = get_current_line(self._text, occ)
                    end_match = re.search(capitals_pattern, end_line)
                    if end_match and (len(end_match.group(0)) / len(end_line) > 0.5):
                        self._end_position = occ
                        return self._end_position
        return None


class SubjectContextEndNameDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # pouze nazev
        article_pattern = 'název'
        start_occurrences = find_all_occurrences_in_string(article_pattern,
                                                           self._text[
                                                           max(start_position - 50, 0):min(start_position + 50,
                                                                                           len(self._text))])
        if len(start_occurrences) > 0:
            end_occurrences = find_all_occurrences_in_string('\n', self._text[start_position:end_position])
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = start_position + occ
                    if len(get_current_line(self._text, occ)) > 30:
                        self._end_position = occ
                        return self._end_position

        return None


class SubjectContextEndQuotationDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        # nazev v uvozovkach
        start_occurrences = find_all_occurrences_in_string('[„"]',
                                                           self._text[
                                                           start_position:min(start_position + 50, len(self._text))])
        if len(start_occurrences) > 0:
            name_start_position = start_position + start_occurrences[0]
            end_occurrences = find_all_occurrences_in_string('["“]', self._text[name_start_position:end_position])
            if len(end_occurrences) > 0:
                self._end_position = name_start_position + end_occurrences[0] + 1
                return self._end_position


class SubjectContextEndWordsDetector(SubjectContextEndDetector):

    def detect(self, start_position, end_position):
        end_words = ['Cena', 'Doba', 'Místo']
        for word in end_words:
            end_occurrences = find_all_occurrences_in_string(word, self._text[start_position:end_position])
            if len(end_occurrences) > 0:
                for occ in end_occurrences:
                    occ = start_position + occ
                    if len(get_current_line(self._text, occ)) < 50:
                        self._end_position = occ
                        return self._end_position


class AdvancedSubjectContextExtractor(SubjectContextExtractor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_subject_context_starts(self, text):
        raw_context_starts = super().get_subject_context_starts(text)
        starts = []
        for start in raw_context_starts:
            end = super().get_subject_context_end(text, start)
            start_occurrences = self.get_all_occurrences(text[start:end])
            df = pandas.DataFrame(start_occurrences, columns=['keyword', 'rat', 'occ'])
            df = df.sort_values(['rat'], ascending=False)
            if len(df.index) > 0:
                start += df.iloc[0]['occ']
            if start not in starts:
                starts.append(start)
        return starts

    def get_subject_context_end(self, text, start_position):
        end_position = SubjectContextEndDetector(text, self._subj_range).detect(start_position)

        detectors = [
            SubjectContextEndNumberingDetector,
            SubjectContextEndRomanNumberingDetector,
            SubjectContextEndHeaderDetector,
            SubjectContextEndCapitalsDetector,
            SubjectContextEndNameDetector,
            SubjectContextEndQuotationDetector,
            SubjectContextEndWordsDetector
        ]

        for detectorCls in detectors:
            detector = detectorCls(text, self._subj_range)
            if detector.detect(start_position, end_position):
                return detector._end_position

        return end_position
