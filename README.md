# TWO PU-learning tools on spark

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/BlancRay/PUAdapter/Scala%20CI?label=Release%20CI&logo=github&style=flat-square)](https://github.com/BlancRay/PUAdapter/actions?query=workflow:"Scala+CI") [![Build Status](https://www.travis-ci.org/BlancRay/PUAdapter.svg?branch=master)](https://www.travis-ci.org/BlancRay/PUAdapter)

## PUAdapter
A set of machine learning tools and algorithms for learning from positive and unlabeled datasets.

This algorithm is based on Positive Samples are Completely Random Selected, and "a classifier trained on positive and
unlabeled examples predicts probabilities that differ by only a constant factor from
the true conditional probabilities of being positive."

Original paper:
>Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data." Proceeding of the 14th
ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.

Code from [pu-learning](https://github.com/aldro61/pu-learning), which is based Python and I transformed code to scala
in order to apply on Spark.

## PU4Spark

A library for Positive-Unlabeled Learning for Apache Spark MLlib (ml package)

Reference to [pu4spark](https://github.com/ispras/pu4spark)

### Implemented algorithms

#### Traditional PU
Original Positive-Unlabeled learning algorithm; firstly proposed in
> Liu, B., Dai, Y., Li, X. L., Lee, W. S., & Philip, Y. (2002).
Partially supervised classification of text documents.
In ICML 2002, Proceedings of the nineteenth international conference on machine learning. (pp. 387–394).

#### Gradual Reduction PU (aka PU-LEA)
Modified Positive-Unlabeled learning algorithm;
main idea is to gradually refine set of positive examples.
Pseudo code was taken from:
>Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015).
Detecting positive and negative deceptive opinions using PU-learning.
Information Processing & Management, 51(4), 433-443.
