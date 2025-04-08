# pychirps
WORK IN PROGRESS

A complete revision of the software I wrote for my PhD: White-boxing Decision Forests (2021). Since the original work, there have been quite a few breaking changes in Pandas. In addition, the original version isn't of a professional standard for offering to the open source community as it was written to support my PhD research and comes with a massive automation framework for running all the experimental research.

This version will solve all of those problems.

## What
SaaS for giving ML predictions (Classification Tasks, primarily) with an explanation.

## How
1. You to upload your tabular data set (csv, etc) and it automatically builds a high performing ML model (Random Forest, AdaBoost or Gradient Boosting) behind the scenes. The model dockerized and deployed.
2. You get end-point, hopefully both a REST API and a GUI where you can send new individual observations. The model will classify the instance and you will get a classification rule side-by-side to show you how the model produced that response.
3. Based on my work in [CHIRPS: Explaining random forest classification](https://link.springer.com/article/10.1007/s10462-020-09833-6), [Ada-WHIPS: explaining AdaBoost classification with applications in the health sciences](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01201-2), and [gbt-HIPS: Explaining the Classifications of Gradient Boosted Tree Ensembles](https://www.mdpi.com/2076-3417/11/6/2511), the classification rule captures the largest region of the feature space that gets the same classification response as your instance, with the smallest number of rule antecedent terms.

### Demos
Requires my data_preprocs library, which I am also currently tweaking occasionally. For ease of development, this library is installed in a local folder which I just update with git pull as appropiate. The dependency is the rebuilt with `make reinstall_data` or:

`poetry remove data_preprocs`

`poetry add --group dev ../data_preprocs`

### TO DO
* Add support for more tree ensemble methods
* Add support for more datasets
* Add support from more sophisticated search/optimization
* Model cache
* Input cache
* Finish and document the APIs