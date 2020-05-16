import re
import time

import ufal.udpipe
from udapi.block.util.filter import Filter
from udapi.core.document import Document

from recommender.component.feature.document import DataProcessor


class TextAnnotator(DataProcessor):
    """Text annotator

    Provides interface to annotate text with UDPipe annotation tool.
    """
    def __init__(self, pipeline='../model/udpipe/udpipe-ud-2.5-191206/czech-pdt-ud-2.5-191206.udpipe', **kwargs):
        """
        Args:
            pipeline: may be one of: path to the UDPipe model, loaded UDPipe model or UDPipe Pipeline itself
        """
        super().__init__(**kwargs)
        if isinstance(pipeline, str):
            self.print('Loading UDPipe model from: ' + pipeline)
            start = time.time()
            pipeline = ufal.udpipe.Model.load(pipeline)
            end = time.time()
            self.print('Model loaded in: ' + str(end - start) + ' sec', 'debug')
        if isinstance(pipeline, ufal.udpipe.Model):
            pipeline = ufal.udpipe.Pipeline(pipeline, "tokenize",
                                            ufal.udpipe.Pipeline.DEFAULT,
                                            ufal.udpipe.Pipeline.DEFAULT, "conllu")
        if not isinstance(pipeline, ufal.udpipe.Pipeline):
            raise ValueError('pipeline must be ' + ufal.udpipe.Pipeline + ' or a path to the UDPipe model')
        self._pipeline = pipeline

    def annotate_text(self, text):
        """Runs annotation of text

        Args:
            text (str): text to be annotated

        Returns:
            str: annotated text
        """
        return self._pipeline.process(text)

    def _process_inner(self, text):
        return self.annotate_text(text)


class UdapiFromConlluTransformer(DataProcessor):
    """Udapi from conllu transformer

    Transforms conllu formated text to Udapi representation.
    """

    @staticmethod
    def from_connlu(conllu):
        """Creates Udapi document and initializes it with conllu string.

        Args:
            conllu (str): conllu formatted string

        Returns:
            Udapi document
        """
        doc = Document()
        doc.from_conllu_string(conllu)
        return doc

    def _process_inner(self, conllu):
        return UdapiFromConlluTransformer.from_connlu(conllu)


class UdapiToStrTransformer(DataProcessor):
    """Udapi to string transformer"""

    @staticmethod
    def to_string(document):
        """Merges whole Udapi document to text string

        Args:
            document: Udapi document

        Returns:
            str: Udapi document converted to string
        """
        lines = []
        for bundle in document.bundles:
            for tree in bundle.trees:
                lines.append(tree.compute_text())
        text = '\n'.join(lines)
        return text

    def _process_inner(self, document):
        return UdapiToStrTransformer.to_string(document)


class UdapiTransformer:
    """Abstract Udapi transformer"""

    @classmethod
    def _iter_sentences(cls, document):
        sentences = []
        for b in document.bundles:
            if not b.trees:
                continue
            sentences.append(b.trees[0])
        return sentences


class UdapiWordOccurrenceSentenceFilter(UdapiTransformer):
    """Udapi word occurrence sentence filter

    Filters out udapi document sentence containing keywords.
    """
    _keywords = None
    _udapi_filters = None

    def __init__(self, keywords=['cena', 'hodnota'], udapi_filters=None):
        self._keywords = keywords
        self._udapi_filters = udapi_filters \
            if udapi_filters is not None else \
            [Filter(delete_tree_if_node='node.lemma=="' + keyword + '"') for keyword in self._keywords]

    def process(self, conllu_document):
        for udapi_filter in self._udapi_filters:
            udapi_filter.process_document(conllu_document)
        return conllu_document


class UdapiWordOccurrencePartSentenceFilter(UdapiTransformer):
    """Udapi word occurrence part sentence filter

    Filters out udapi document sub-sentence containing keywords.
    """
    _keywords = None
    _udapi_filters = None

    def __init__(self, keywords=['cena', 'hodnota'], udapi_filters=None):
        self._keywords = keywords
        self._udapi_filters = udapi_filters \
            if udapi_filters is not None else \
            [Filter(delete_subtree='node.lemma=="' + keyword + '"') for keyword in self._keywords]

    def process(self, conllu_document):
        for udapi_filter in self._udapi_filters:
            udapi_filter.process_document(conllu_document)
        return conllu_document


class NodeFinder:
    """Udapi document node finder

    Provides interface for searching through Udapi document.
    """
    def __init__(self, feat_val_pairs, max_dist=100, fnext_node=lambda n: n.next_node, check_current=False):
        """
        Args:
            feat_val_pairs: specification of condition of searched features and theirs values
            max_dist (int): maximum searched distance from origin node
            fnext_node (func): function defining next node
            check_current (bool): whether to check origin node or not
        """
        self._feat_val_pairs = feat_val_pairs if isinstance(feat_val_pairs, list) else [feat_val_pairs]
        self._feat_val_pairs = [p if isinstance(p, list) else [p] for p in self._feat_val_pairs]

        for i, ors in enumerate(self._feat_val_pairs):
            self._feat_val_pairs[i] = [(feat, vals if isinstance(vals, list) else [vals])
                                       for feat, vals in ors]

        for i, ors in enumerate(self._feat_val_pairs):
            for j, ands in enumerate(ors):
                patterns = [re.compile('^' + pattern + '$') for pattern in ands[1]]
                ors[j] = (ands[0], patterns)

        self._max_dist = max_dist
        self._fnext_node = fnext_node
        self._check_current = check_current

    def _check_disjunctive_node_attributes(self, node, disjunctions):
        for feat, patterns in disjunctions:
            try:
                attr = node._get_attr(feat)
            except AttributeError:
                continue
            if not any(pat.search(attr) for pat in patterns):
                return False
        return True

    def check_node_attributes(self, node):
        """Check the target features

        Args:
            node: Udapi node to be checked

        Returns:
            bool: True if any of the features values is corresponding to the nodes values.
        """
        if any(self._check_disjunctive_node_attributes(node, ors) for ors in self._feat_val_pairs):
            return True
        return False

    def _find_internal(self, origin):
        node = origin
        dist = 0
        while dist < self._max_dist:
            node = self._fnext_node(node)
            if not node or node.is_root():
                break
            dist += 1
            if self.check_node_attributes(node):
                return node
        return None

    def find(self, origin):
        """Runs the search from origin node

        Args:
            origin: Udapi node to start the search from

        Returns:
            first Udapi node that matches the condition, or None
        """
        if self._check_current and self.check_node_attributes(origin):
            return origin
        return self._find_internal(origin)


class NextNodeFinder(NodeFinder):
    """Udapi document next node finder"""
    def __init__(self, feat_val_pairs, max_dist=100, check_current=False):
        super().__init__(feat_val_pairs, max_dist, lambda n: n.next_node, check_current)


class PreviousNodeFinder(NodeFinder):
    """Udapi document previous node finder"""
    def __init__(self, feat_val_pairs, max_dist=100, check_current=False):
        super().__init__(feat_val_pairs, max_dist, lambda n: n.prev_node, check_current)


class PrecedingNodeFinder(NodeFinder):
    """Udapi document preceding node finder"""
    def __init__(self, feat_val_pairs, max_depth=100, check_current=False):
        super().__init__(feat_val_pairs, max_depth, lambda n: n.parent, check_current)


class DescendingNodeFinder(NodeFinder):
    """Udapi document descending node finder"""
    def __init__(self, feat_val_pairs, max_depth=100, check_current=False):
        super().__init__(feat_val_pairs, max_depth, None, check_current)

    def _find_internal(self, origin):
        orig_depth = origin._get_attr('depth')
        for node in origin.descendants:
            if orig_depth - node._get_attr('depth') <= self._max_dist:
                if self.check_node_attributes(node):
                    return node
        return None


class SubmitterPartSentenceFilter(UdapiTransformer):
    """Udapi submitter part sentence filter

    Filters out udapi document sub-sentence submitter reference.
    """
    _submitter_keywords = None
    _suplier_keywords = None

    def __init__(self, submitter_keywords=['objednatel', 'zadavatel', 'kupující'],
                 suplier_keywords=['dodavatel', 'prodávající', 'zhotovitel', 'uchazeč']):
        self._submitter_keywords = submitter_keywords
        self._suplier_keywords = suplier_keywords

    def process(self, document):
        for tree in self._iter_sentences(document):
            for n in tree.descendants:
                if n.lemma.lower() in ['objednatel', 'kupující', 'zadavatel'] and n.feats['Case'] == 'Nom':
                    p = PrecedingNodeFinder(('upos', 'VERB')).find(n)
                    if not p:
                        p = n
                    for n2 in p.descendants:
                        if n2.lemma.lower() in ['dodavatel', 'prodávající', 'zhotovitel', 'uchazeč'] \
                                and n2.feats['Case'] == 'Nom':
                            p2 = p = PrecedingNodeFinder(('upos', 'VERB')).find(n2)
                            if not p2:
                                p2 = n2
                            p2.parent = p.parent
                    p.remove()
                    tree.text = tree.compute_text()
        return document


class NonSubjectPartSentenceFilter(UdapiTransformer):
    """Non subject part sentence filter

    Filters out part of sentences that does not suit a complex condition.

    The complex condition is combined of:
        target dependency relations,
        target verb lemmas,
        banned UPOS tags,
        banned node lemmas,
        banned submitter lemmas,
        banned location and time lemmas,
        banned previous lemmmas,
        banned preceding lemmas
    """
    def __init__(self,
                 target_dep_relations=['nsubj', 'nsubj:pass', 'obj', 'obl', 'obl:arg', 'nmod'],
                 target_verb_lemmas=['zavazovat', 'doda(t|ný)', 'zajistit', 'prov(edený|ést)',
                                     'zahrnovat', 'spočíva(t|jící)', 'rozumět'],
                 banned_upos_tags=['PUNCT', 'SYM', 'NUM', 'DET'],
                 banned_node_lemmas=['smlouva', 'předmět', 'uzavření', 'náklad', 'nebezpečí', 'specifikace',
                                     'dodavatel', 'závazek', 'dokumentace', 'rozsah', 'plnění', 'zakázka',
                                     'dohoda', 'poplatek', 'požádání', 'záměr', 'dodatek', 'podmínka', 'standard',
                                     'norma', 'kvalita', 'prohlášení', 'jiný', 'osoba', 'souhrn'],
                 banned_submitter_lemmas=['objednatel', 'kupující', 'zadavate[lt]', 'projektant'],
                 banned_loctime_lemmas=['místo', 'doba'],
                 previous_banned_lemmas=['zbytný', 'pokud', 'dodatečný', 'dokumentace', 'nutný'],
                 preceding_banned_lemmas=['specifikovaný', 'povinný', 'oprávněný', 'možný', 'uvedený']):

        self._target_dep_relations = target_dep_relations
        self._target_verb_lemmas = target_verb_lemmas
        self._banned_upos_tags = banned_upos_tags
        self._banned_node_lemmas = banned_node_lemmas

        self._banned_submitter_lemmas = banned_submitter_lemmas
        self._banned_loctime_lemmas = banned_loctime_lemmas
        self._previous_banned_lemmas = previous_banned_lemmas
        self._preceding_banned_lemmas = preceding_banned_lemmas

        self._target_deprel_checker = NodeFinder(('deprel', self._target_dep_relations))
        self._banned_upos_checker = NodeFinder([('upos', self._banned_upos_tags), ('feats[Case]', 'Ins')])
        self._banned_lemmas_checker = NodeFinder(('lemma', self._banned_node_lemmas))

        self._banned_submitter_lemma_finder = PreviousNodeFinder(('lemma', self._banned_submitter_lemmas), 10, True)
        self._banned_loctime_lemma_finder = PreviousNodeFinder(('lemma', self._banned_loctime_lemmas), 10, True)
        self._previous_banned_lemma_finder = PreviousNodeFinder(('lemma', self._previous_banned_lemmas), 5)
        self._preceding_banned_lemma_finder = PrecedingNodeFinder(('lemma', self._preceding_banned_lemmas), 4)
        self._previous_target_verb_finder = PreviousNodeFinder([('lemma', self._target_verb_lemmas),
                                                                [('lemma', 'být'), ('deprel', ['cop', 'root', 'acl']),
                                                                 ('feats[Polarity]', 'Pos')]], 5)

    def _get_candidate_nodes(self, tree):
        candidates = []
        for n in tree.descendants:

            if not self._target_deprel_checker.check_node_attributes(n):
                continue

            if self._banned_upos_checker.check_node_attributes(n):
                continue

            if self._banned_lemmas_checker.check_node_attributes(n):
                continue

            n2 = self._banned_submitter_lemma_finder.find(n)
            if n2 and (n2 == n or not PreviousNodeFinder(('lemma', 'pro'), 2).find(n2)):
                continue

            n2 = PreviousNodeFinder(('lemma', ['dílo']), 5, True).find(n)
            if n2 and (n2 == n or not (NextNodeFinder(('upos', 'DET'), 5).find(n2) or
                                       PreviousNodeFinder(('lemma', 'předmět'), 2).find(n2))):
                continue

            if self._banned_loctime_lemma_finder.find(n):
                continue

            if self._previous_banned_lemma_finder.find(n):
                continue

            if self._preceding_banned_lemma_finder.find(n):
                continue

            n2 = PreviousNodeFinder(('lemma', ['proveden(í|ý)']), 5).find(n)
            if n2 and PreviousNodeFinder([('lemma', 'předmět'), ('upos', 'DET')], 3).find(n2):
                candidates.append(n)
                continue

            if NodeFinder(('deprel', ['nmod'])).check_node_attributes(n):
                if PrecedingNodeFinder(('lemma', ['pořízení', 'uzavřen(í|ý)', 'zabezpečení'])).find(n):
                    candidates.append(n)
                continue

            if not self._previous_target_verb_finder.find(n):
                continue

            if NodeFinder(('deprel', ['obl', 'obl:arg'])).check_node_attributes(n):
                if not NodeFinder(('lemma', ['realizace'])).check_node_attributes(n):
                    if not PreviousNodeFinder(('lemma', ['předmět']), 5).find(n):
                        candidates.append(n)
                continue

            # deprel: nsubj, obj
            candidates.append(n)

        return candidates

    def _extend_candidates(self, candidates):
        # doplnim konjunkce
        extended_candidates = []
        i = 0
        while True:
            if not len(candidates) > i:
                break
            n = candidates[i]
            i += 1
            extended_candidates.append(n)
            # if not (extended_candidates and n.is_descendant_of(extended_candidates[-1])):
            #     extended_candidates.append(n)
            # else:
            #     n = extended_candidates[-1]
            if n.descendants and n.descendants[-1].next_node != n:
                n = n.descendants[-1]
            while True:
                n = NextNodeFinder(('deprel', ['conj', 'nsubj', 'obj']), 7).find(n)
                if not n or (len(candidates) > i and n.is_descendant_of(candidates[i])):
                    break
                extended_candidates.append(n)
        return extended_candidates

    def _get_all_node_candidates(self, node):
        tmp_candidates = []
        # ponecham jen slova za pripadnym nasledujicim slovesem
        n = DescendingNodeFinder([('upos', 'VERB'), ('deprel', 'cop')]).find(node)
        if n and abs(n.ord - node.ord) < 10 \
                and NodeFinder(('form', '(je|.{3,})')).check_node_attributes(n) \
                and not PreviousNodeFinder(('upos', 'DET'), 2).find(n):
            # abs(n2.ord - n.ord) < 10 kvuli bugu s nespravne tokenizovanzmi vetami
            # (je|.{3,}) kvuli bugu s identifikaci slovesa na nerelevantnich tokenech
            n = n.next_node
            while n and (n.is_descendant_of(node) or n == node):
                tmp_candidates.append(n)
                n = n.next_node
        else:
            tmp_candidates.append(node)
            tmp_candidates.extend(node.descendants)
            # odříznu příliš dlouhé řetězce
        keep_node_candidates = []
        for n in tmp_candidates:
            subj_node = PrecedingNodeFinder(('lemma', 'předmět')).find(n)
            if not (subj_node and subj_node.is_descendant_of(node)):
                if abs(n.ord - node.ord) > 10 and n.upos == 'PUNCT':
                    break
                keep_node_candidates.append(n)
        return keep_node_candidates

    def process(self, document):
        """Runs the filtering process for Udapi document

        Identifies candidate nodes for subject information using the complex condition.
        Extends the candidates with their conjunctions.
        Extends the candidates with relevant subtrees.
        Removes non-candidate nodes.

        Args:
            document: Udapi document to be filtered

        Returns:
            filtered Udapi document
        """
        for i2, tree in enumerate(self._iter_sentences(document)):
            if not tree.children:
                continue

            candidates = self._get_candidate_nodes(tree)
            candidates = self._extend_candidates(candidates)

            keep_nodes = []
            for n in candidates:
                node_candidates = self._get_all_node_candidates(n)
                keep_nodes.extend(node_candidates)
            keep_nodes = [n for n in keep_nodes
                          if not NodeFinder(('lemma', ['smlouva', 'uzavření', 'předmět', 'plnění']))
                            .check_node_attributes(n)]

            to_remove = [n for n in tree.descendants if not n in keep_nodes]

            for n in keep_nodes:
                n.parent = tree
            for n in to_remove:
                n.remove()

            tree.text = tree.compute_text()
        return document


class EmptyBundlesFilter:
    """Empty bundles filter

    Removes empty sentences from document.
    """

    def process(self, document):
        bundles = []
        for b in document.bundles:
            if not b.trees:
                continue
            if not b.trees[0]:
                continue
            if not b.trees[0].descendants:
                continue
            bundles.append(b)
        document.bundles = bundles
        return document


class ConlluSubjectContextPreprocessor(DataProcessor):
    """Conllu subject context preprocessor

    Uses specific Conllu filters to process the text.
    """
    _transformers = None

    def __init__(self, transformers=None):
        """
        Args:
            transformers (list): list of specific Conllu (Udapi) filters
        """
        super().__init__()
        self._transformers = transformers \
            if transformers is not None else \
            [
                UdapiWordOccurrencePartSentenceFilter(keywords=['cena', 'hodnota', 'DPH']),
                UdapiWordOccurrencePartSentenceFilter(keywords=['příloha', 'dále', 'jen']),
                NonSubjectPartSentenceFilter(),
                EmptyBundlesFilter(),
            ]

    def transform_document(self, conllu_document):
        """Runs the processing of document.

        Runs the processing of all transformers one by one.

        Args:
            conllu_document: Udapi document to be processed

        Returns:
            processed Udapi document
        """
        for transformer in self._transformers:
            conllu_document = transformer.process(conllu_document)
        return conllu_document

    def _process_inner(self, conllu_document):
        return self.transform_document(conllu_document)
