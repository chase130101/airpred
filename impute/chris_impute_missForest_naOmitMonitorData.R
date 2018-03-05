require('missForest')

# read in data
sensor.census.to.impute = read.csv('subet_sensor_census_readyToImpute.csv')

# imputing with random forest
sensor.census.imputed.rf = missForest(sensor.census.to.impute, maxiter = 10, ntree = 1, variablewise = T,
                                      decreasing = F, verbose = T, mtry = floor(sqrt(ncol(sensor.census.to.impute))), replace = TRUE)

sensor.census.imputed.rf