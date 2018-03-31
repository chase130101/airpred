# read in data
sensor_census = read.csv('./subset_sensor_census_toImpute.csv')

# get correlation matrix with for all variables
corr.mat = as.matrix(cor(sensor_census, use = 'pairwise.complete.obs')) - diag(ncol(sensor_census))

# get variable names for variables with high correlations 
high.corr.colnames = colnames(corr.mat)[apply(corr.mat, 1, max) >= 0.9]

# columns to delete
to.delete = c('USElevation_dsc10000','USElevation_max100','USElevation_max10000','USElevation_mea10000','USElevation_med100','USElevation_med10000','USElevation_min100',
              'USElevation_min10000','USElevation_bln100','USElevation_bln10000', 'NLCD_Developed10000', 'NLCD_Impervious10000', 'MAIACUS_Optical_Depth_055_Aqua_Nearest4', 
              'MAIACUS_Optical_Depth_055_Terra_Nearest4', 'REANALYSIS_shum_2m_DailyMax', 'REANALYSIS_prate_DailyMax', 'REANALYSIS_prate_DailyMean', 'REANALYSIS_dlwrf_DailyMean',
              'REANALYSIS_shum_2m_DailyMin', 'REANALYSIS_shum_2m_1Day', 'REANALYSIS_air_sfc_DailyMin', 'REANALYSIS_air_sfc_DailyMax', 'REANALYSIS_air_sfc_DailyMean',
              'Nearby_Peak2_MaxTemperature', 'Nearby_Peak2_MinTemperature', 'Nearby_Peak2Lag1_MaxTemperature', 'Nearby_Peak2Lag1_MeanTemperature', 'Nearby_Peak2Lag1_MinTemperature', 
              'Nearby_Peak2Lag3_MaxTemperature', 'Nearby_Peak2Lag3_MinTemperature')

# columns to keep that have high correlations by initial analysis
high.corr.colnames.del = high.corr.colnames[!(high.corr.colnames %in% to.delete)]

# to inspect which variables have high correlations
#corr.mat[high.corr.colnames.del, high.corr.colnames.del]

# all columns to keep
cols.to.keep = names(sensor_census)[!(names(sensor_census) %in% to.delete)]

# final data for imputation
sensor_census = sensor_census[, cols.to.keep]

# delete rows with monitor data NA
sensor_census = sensor_census[!is.na(sensor_census$MonitorData),]

# get dataframe with site and monitor data
site_MonitorData = sensor_census[, c('site', 'MonitorData')]

sensor_census = sensor_census[, !(names(sensor_census) %in% c('site', 'MonitorData'))]

# imputing with random forest
require('missForest')
require('doParallel')
registerDoParallel(cores = 10) 
sensor_census_imputed_rf = missForest(sensor_census, maxiter = 10, ntree = 10, variablewise = T,
                                      decreasing = F, verbose = T, mtry = floor(sqrt(ncol(sensor_census))), replace = TRUE, parallelize = 'variables')

# add site and MonitorData back in
sensor_census_imp = cbind(site_MonitorData, sensor_census_imputed_rf$ximp)

# write to csv for modeling in python
write.csv(sensor_census_imp, './', row.names = F)


