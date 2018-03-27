#Load packages
library(dplyr)


#Read data
pollution_data = read.csv('random_subset.csv')
location_census_data = read.csv('sensor_locations_with_census.csv')

#Join census data with pollution data
full_data = left_join(pollution_data, location_census_data, by = 'site')

#Join sensors outside of the continental United States
full_data = filter(full_data, Continental_indicator == 1)

#Convert date to date format
full_data$date = as.Date(full_data$date)

#Extract month from date
full_data$month = as.numeric(format(full_data$date,"%m"))

#Move month variable to the front of the data frame
full_data = full_data %>% select(site, year, month, everything())

#Redefine year as number of years from starting point
full_data$year = full_data$year - min(full_data$year)

#Create latitude, longitude interaction variable 
full_data$Lat_Lon_int = full_data$Lat * full_data$Lon 

#Variables to drop from data frame
variables_to_drop = c('White', 'Black', 'Native', 'Asian', 'Islander', 'Other', 'Two', 'Hispanic', 
                      'Age_0_to_9', 'Age_10_to_19', 'Age_20_to_29','Age_30_to_39',
                      'Age_40_to_49','Age_50_to_59','Age_60_to_69', 'Age_70_plus', 
                      'Income_less_than_25k', 'Income_25k_to_50k', 'Income_25k_to_50k', 'Income_50k_to_75k', 
                      'Income_75k_to_100k', 'Income_100k_to_150k', 'Income_150k_to_200k', 'Income_200k_or_more',
                      'Households', 'Family_Households', 'City', 'State', 'County', 'Zip', 'Country', 
                      'Continental_indicator', 'REANALYSIS_windspeed_10m_1Day', 
                      'USElevation_dsc10000','USElevation_max100','USElevation_max10000','USElevation_mea10000',
                      'USElevation_med100','USElevation_med10000','USElevation_min100','USElevation_min10000',
                      'USElevation_bln100','USElevation_bln10000', 'NLCD_Developed10000', 'NLCD_Impervious10000', 
                      'MAIACUS_Optical_Depth_055_Aqua_Nearest4','MAIACUS_Optical_Depth_055_Terra_Nearest4', 
                      'REANALYSIS_shum_2m_DailyMax', 'REANALYSIS_prate_DailyMax', 'REANALYSIS_prate_DailyMean', 
                      'REANALYSIS_dlwrf_DailyMean','REANALYSIS_shum_2m_DailyMin', 'REANALYSIS_shum_2m_1Day', 
                      'REANALYSIS_air_sfc_DailyMin', 'REANALYSIS_air_sfc_DailyMax', 'REANALYSIS_air_sfc_DailyMean',
                      'Nearby_Peak2_MaxTemperature', 'Nearby_Peak2_MinTemperature', 'Nearby_Peak2Lag1_MaxTemperature', 
                      'Nearby_Peak2Lag1_MeanTemperature', 'Nearby_Peak2Lag1_MinTemperature', 
                      'Nearby_Peak2Lag3_MaxTemperature', 'Nearby_Peak2Lag3_MinTemperature')

#Remove variables to drop from data
full_data = select(full_data, -one_of(variables_to_drop))

#This is only needed if the data is not sorted
#full_data = full_data %>% arrange(site, date)

#Create vector of all site ids
sites = unique(full_data$site)
n = length(sites)

#Sites for train data
train_sites = sort(sample(sites, ceiling(.8*n)))

#Split data into train and test
train_data = full_data %>% filter(site %in% train_sites) 
test_data = full_data %>% filter(!(site %in% train_sites))

#Split test and train into predictors and non-predictors
x_train = select(train_data, -c(site, date, MonitorData))
y_train = select(train_data, c(site, date, MonitorData))

x_test = select(test_data, -c(site, date, MonitorData))
y_test = select(test_data, c(site, date, MonitorData))

#Impute missing values with mean
#Impute train predictors 

for(i in 1:ncol(x_train)){
  
  x_train[is.na(x_train[,i]), i] = mean(x_train[,i], na.rm = T)
  
}
imputed_train_mean = cbind(y_train, x_train)

#Attach imputed train predictors to test predictors, then impute test predictors 
x_test_with_imputed_train = rbind(x_train, x_test)

for(i in 1:ncol(x_test_with_imputed_train)){
  
  x_test_with_imputed_train[is.na(x_test_with_imputed_train[,i]), i] = mean(x_test_with_imputed_train[,i], na.rm = T)

}

#Removed previously attached train data
imputed_xtest_mean = x_test_with_imputed_train[-(1:nrow(x_train)),]
imputed_test_mean = cbind(y_test, imputed_xtest_mean)

#Write to csv for modeling in python
write.csv(imputed_train_mean, '../data/imputed_train_mean.csv', row.names = F)
write.csv(imputed_test_mean, '../data/imputed_test_mean.csv', row.names = F)
