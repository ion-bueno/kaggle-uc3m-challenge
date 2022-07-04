library('missRanger')
library(dplyr)

data = read.csv('../raw_data/train_set.csv')

dataImputed = missRanger(data, pmm.k = 10, num.trees = 100)

write.csv(dataImputed, '../preprocessed/data_pmm.csv')
