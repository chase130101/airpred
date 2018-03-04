require('mice')
require('ImputeRobust')
require('lattice')

# read in data
sensor.census.to.impute = read.csv('subet_sensor_census_readyToImpute.csv')
#sensor.census.to.impute2 = sensor.census.to.impute[, -((ncol(sensor.census.to.impute)-11):ncol(sensor.census.to.impute))]


# Create the imputed data sets
imputed.sets = mice(sensor.census.to.impute[1:50000, ], m = 1, maxit = 1,
                    method = rep('rf', ncol(sensor.census.to.impute)),
                    visitSequence = 'monotone',
                    seed = 97123)
