from unittest import TestCase

import numpy as np
from sklearn import feature_extraction as sklearn_feature_extraction

from mla_sani import feature_extraction

class testDictVectorizer(TestCase):
    def test_transform(self):
        X = [
            {'city': 'Dubai', 'temperature': 33.},
            {'city': 'London', 'temperature': 12.},
            {'city': 'San Francisco', 'temperature': 18.}
        ]
        vec1 = feature_extraction.DictVectorizer()
        X_trans1 = vec1.fit_transform(X)

        vec2 = sklearn_feature_extraction.DictVectorizer()
        X_trans2 = vec2.fit_transform(X).toarray()

        self.assertTrue(np.allclose(X_trans1, X_trans2))
        self.assertListEqual(vec1.feature_names_, vec2.feature_names_)
        self.assertDictEqual(vec1.vocabulary_, vec2.vocabulary_)


class testTfidfTransformer(TestCase):
    def test_transform(self):
        counts = np.array([
            [3, 0, 1],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [3, 2, 0],
            [3, 0, 2]
        ])

        tfidf1 = feature_extraction.TfidfTransformer
        tfidf2 = sklearn_feature_extraction.text.TfidfTransformer

        self.assertTrue(np.allclose(tfidf1(norm="l1", use_idf=True, smooth_idf=True).fit_transform(counts),
                                    tfidf2(norm="l1", use_idf=True, smooth_idf=True).fit_transform(counts).toarray()))

        self.assertTrue(np.allclose(tfidf1(norm="l2", use_idf=True, smooth_idf=True).fit_transform(counts),
                                    tfidf2(norm="l2", use_idf=True, smooth_idf=True).fit_transform(counts).toarray()))


        self.assertTrue(np.allclose(tfidf1(norm="l1", use_idf=False, smooth_idf=True).fit_transform(counts),
                                    tfidf2(norm="l1", use_idf=False, smooth_idf=True).fit_transform(counts).toarray()))


        self.assertTrue(np.allclose(tfidf1(norm="l2", use_idf=True, smooth_idf=False).fit_transform(counts),
                                    tfidf2(norm="l2", use_idf=True, smooth_idf=False).fit_transform(counts).toarray()))
