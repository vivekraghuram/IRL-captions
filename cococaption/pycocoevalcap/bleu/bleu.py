#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .bleu_scorer import BleuScorer
import numpy as np


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        bleu_scorer = BleuScorer(n=self._n)
        hypo_len = len(res[imgIds[0]])
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == hypo_len)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            for i in range(hypo_len):
              bleu_scorer += (hypo[i], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        # return score, scores
        scores = np.array(scores)
        scores = scores.reshape((hypo_len * len(imgIds), 4))
        return scores[:, -1]

    def method(self):
        return "Bleu"
