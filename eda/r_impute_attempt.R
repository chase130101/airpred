require('mice')

# read in data
sensor.census.to.impute = read.csv('subet_sensor_census_readyToImpute.csv')

# Create the imputed data sets
imputed.sets = mice(sensor.census.to.impute[1:10000,], m = 1, maxit = 1,
                    method = 'mean', ntree = 10,
                    visitSequence = 'monotone',
                    seed = 97123)

missForest(sensor.census.to.impute[1:10000,], maxiter = 10, ntree = 10, variablewise = T,
           decreasing = F, verbose = T,
           mtry = floor(sqrt(ncol(sensor.census.to.impute))), replace = TRUE)