#Load packages
require(dplyr)
require(data.table)


#Read data
#pollution_data = fread('../data/random_subset_1p.csv')
pollution_data = readRDS('assembled_data.Rds')
location_census_data = fread('../data/sensor_locations_with_census.csv')

#Join census data with pollution data
full_data = left_join(pollution_data, location_census_data, by = 'site')

#Remove sensors outside of the continental United States
full_data = filter(full_data, Continental_indicator == 1)

#Convert date to date format
full_data$date = as.Date(full_data$date)

#Extract month from date
full_data$month = as.numeric(format(full_data$date,"%m"))

#Redefine year as number of years from starting point
full_data$year = full_data$year - min(full_data$year)

#Engineer other date features
full_data$cumulative_month = full_data$year*12 + full_data$month 
full_data$day_of_year = as.numeric(format(full_data$date,"%d")) + 30*(full_data$month-1)
full_data$cumulative_day =  365*(full_data$year) + full_data$day_of_year
full_data$month = as.factor(full_data$month)

#Move date variables to the front of the data frame
full_data = full_data %>% select(site, year, month, cumulative_month,
                                 day_of_year, cumulative_day, everything())

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
                      'Nearby_Peak2Lag3_MaxTemperature', 'Nearby_Peak2Lag3_MinTemperature',
                      'Islander_p', 'Other_p', 'Two_p')

#Remove variables to drop from data
full_data = select(full_data, -one_of(variables_to_drop))

#Remove variables that we do not want to impute
impute_variables = select(full_data, -c(site, date, month, MonitorData))
other_variables = select(full_data, c(site, date, month, MonitorData))


#Function to impute the mean 
impute_mean = function(x){
  replace(x, is.na(x), mean(x, na.rm = T))
}

#Impute missing values with mean
for(i in 1:ncol(impute_variables)){
  impute_variables[,i] = impute_mean(impute_variables[,i])
}

#Merge previously removed variables
imputed_data_mean = cbind(other_variables, impute_variables)


#Write to csv for modeling in python
fwrite(imputed_data_mean, '../data/imputed_data_mean.csv')

print('Done')