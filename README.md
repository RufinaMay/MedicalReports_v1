# MedicalReports_v1.0

## Data set
Indiana University Chest X-ray collection is used in this work. There are total 7,470 samples in the data set and 559 unique tags. Data was split on train, test and validation sets using stratified data split [On the stratification of multi-label data](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10). The data is stored in the following format \textit{path to image}-[list of tags] and located in three following files:
- test_set.pickle - 471 unique tags
- valid_set.pickle - 372 unique tags
- train_set.pickle - 590 unique tags
