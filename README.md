# PUAdapter
## A PU-learning tool on spark

Based on Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data." Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.

Code from [pu-learning](https://github.com/aldro61/pu-learning), which is based Python and I transform code to scala in order to apply on Spark.

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
Pseudocode was taken from:
>Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015).
Detecting positive and negative deceptive opinions using PU-learning.
Information Processing & Management, 51(4), 433-443.
