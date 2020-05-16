import re
import pandas
import numpy
import random

from recommender.component.base import DataProcessor

from recommender.component.feature.document import find_all_occurrences_in_string, get_current_line, flatten_column, \
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


class SubjectContextExtractor(DataProcessor):
    """Basic subject context extractor

    Provides the extraction of the subject context parts of documentation.
    Uses weighted keywords for identification of subject context part.
    """
    def __init__(self, keywords=DEF_KEYWORDS, subj_range=2040, **kwargs):
        """
        Args:
            keywords (dict of str: int): keywords specification for identifying the subject context
            subj_range (int): default range of the subject context in number of chars
        """
        super().__init__(**kwargs)
        self._keywords = keywords
        self._subj_range = subj_range

    def get_all_occurrences(self, text):
        """Finds all occurrences of keywords and computes their rating.

        Uses member keywords with their weights to initialize the rating of each of them.
        Uses local characteristics to accumulate the rating coefficient.
        Result keyword rating is the default rating multiplied by the coefficient.

        Args:
            text (str): text to find the keywords in

        Returns:
            list: list of all occurrences represented by a dictionary
            containing the keyword, rating and occurrence position
        """
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
                # Simple sentences following
                koef += chars_occurrence_ratio(text[min(o + 50, len(text)): min(o + 100, len(text))])
                # Nearly noun 'Zbozi' after the pattern ()
                if 'Zboží' in text[o: min(o + 150, len(text))]:
                    koef *= 0.5

                rat *= koef
                occurrences.append({'keyword': matched, 'rat': rat, 'occ': o})
        return occurrences

    def get_subject_context_old(self, text):
        """Gets the context of fixed range located by the occurrence with the highest rating.

        Args:
            text (str): text to find the context in

        Returns:
            str: found context
        """
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
        """Gets subject contexts starts.

        Uses complex algorithm to identify the most relevant subject context parts starts.

        Args:
            text (str): text to find the starts in

        Returns:
            list of in: list of identified subject contexts starts
        """
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
        """Computes the subject context end regarding its start and fixed range.

        Args:
            text (str): text to identify the end in
            start (int): start position of the subject context

        Returns:
            int: position of the identified subject context end
        """
        return min(start + self._subj_range, len(text))

    def get_subject_context_ends(self, text, starts):
        """Computes the subject contexts ends.

        Args:
            text (str): text to identify ends in
            starts (list of int): start positions of the subject contexts

        Returns:
            int: position of the identified subject context end
        """
        return [self.get_subject_context_end(text, start) for start in starts]

    def get_subject_contexts(self, text):
        """Gets multiple contexts located by subject contexts starts and subject contexts ends.

        Args:
            text (str): text to find the contexts in

        Returns:
            list of str: list of found contexts
        """
        starts = self.get_subject_context_starts(text)
        ends = self.get_subject_context_ends(text, starts)
        subj_contexts = [text[start:end] for start, end in zip(starts, ends)]
        return subj_contexts

    def show_occurrence_distribution(self, text, figsize=(15, 5)):
        """Plots the distribution of keyword occurrences

        Args:
            text (str): text to find the occurrences in
            figsize (tuple): size of the figure
        """
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

        ticks = [str(b) + ' (' + str(s) + ')' for b, s in zip(top_bins, [hist[0][i] for i in top_bins])]
        ax = pandas.DataFrame(hist[0]).plot.bar(figsize=figsize, xticks=top_bins)
        ax.set_title('Vážený histogram výskytů klíčových slov')
        ax.set_xticklabels(ticks, rotation=45)
        ax.legend(['vážené ohodnocení binů'])
        ticks = [str(b) + ' (' + str(s) + ')' for b, s in zip(top_bins, [score[i] for i in top_bins])]
        ax = pandas.DataFrame(score).plot.bar(figsize=(figsize[0], figsize[1] * 1.5), xticks=top_bins)
        ax.set_title('Skóre výskytů po provedení konvoluce')
        ax.set_xticklabels(ticks, rotation=45)
        ax.legend(['výsledné skóre binů'])

    def _process_inner(self, text):
        return self.get_subject_contexts(text)


class SubjectContextEndDetector:
    """Basic context end detector"""
    def __init__(self, text, subj_range=2040, multip=3):
        """
        Args:
            text (str): text to detect the context end in
            subj_range (int): default range of the subject context in number of chars
            multip (int): multiplication coefficient of base subject range
        """
        self._text = text
        self._subj_range = subj_range
        self._multip = multip
        self._end_position = None

    def detect(self, start_position, end_position=None):
        """Searches for a new document mark to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
        self._end_position = min(start_position + self._subj_range * self._multip, len(self._text))
        endfile_mark = '<FILE'
        end_occurrences = find_all_occurrences_in_string(endfile_mark, self._text[start_position:self._end_position])
        if len(end_occurrences) > 0:
            self._end_position = start_position + end_occurrences[0]
        return self._end_position


class SubjectContextEndNumberingDetector(SubjectContextEndDetector):
    """Subject context end numbering detector"""

    def detect(self, start_position, end_position):
        """Detects the numbering of sections to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end roman numbering detector"""

    def detect(self, start_position, end_position):
        """Detects the roman numbering of sections to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end header detector"""

    def detect(self, start_position, end_position):
        """Searches for a specific heading keyword of sections to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end upper case detector"""

    def detect(self, start_position, end_position):
        """Detects upper case formatted heading of text to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end name detector"""

    def detect(self, start_position, end_position):
        """Searches for a specific heading keyword of name parameter to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end quotation detector"""

    def detect(self, start_position, end_position):
        """Searches for quotation marks to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Subject context end quotation detector"""

    def detect(self, start_position, end_position):
        """Searches for special keywords to identify the end.

        Args:
            start_position (int): raw start position of the context
            end_position (int): raw end position of the context

        Returns:
            int: position of the identified subject context end
        """
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
    """Advanced subject context extractor

    Extends the algorithm of selecting and computing the subject context starts and ends.
    """
    def __init__(self, max_contexts=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_contexts = max_contexts

    def get_subject_context_starts(self, text):
        """Gets subject contexts starts.

        Extends the algorithm of identifying the starts with detailed aim to the most relevant occurrence.

        Args:
            text (str): text to find the starts in

        Returns:
            list of in: list of identified subject contexts starts
        """
        raw_context_starts = super().get_subject_context_starts(text)
        if len(raw_context_starts) > self._max_contexts:
            raw_context_starts = random.sample(raw_context_starts, self._max_contexts)
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
        """Computes the relevant subject context end.

        Uses specific detectors to identify relevant end of the context.

        Args:
            text (str): text to identify the end in
            start_position (int): start position of the subject context

        Returns:
            int: position of the identified subject context end
        """
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
