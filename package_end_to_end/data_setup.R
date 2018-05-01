#Load packages
require(data.table)
require(dplyr)

#Read data
#pollution_data = fread('../data/assembled_data.csv')
pollution_data = readRDS('../data/assembled_data.rds')
location_census_data = fread('../data/sensor_locations_with_census.csv')

#Join census data with pollution data
full_data = left_join(pollution_data, location_census_data, by = 'site')

#Remove sensors outside of the continental United States
full_data = filter(full_data, Continental_indicator == 1)

#Variables to drop from data frame
variables_to_drop = c('White', 'Black', 'Native', 'Asian', 'Islander', 'Other', 'Two', 'Hispanic', 
                      'Age_0_to_9', 'Age_10_to_19', 'Age_20_to_29','Age_30_to_39',
                      'Age_40_to_49','Age_50_to_59','Age_60_to_69', 'Age_70_plus', 
                      'Income_less_than_25k', 'Income_25k_to_50k', 'Income_25k_to_50k', 'Income_50k_to_75k', 
                      'Income_75k_to_100k', 'Income_100k_to_150k', 'Income_150k_to_200k', 'Income_200k_or_more',
                      'Households', 'Family_Households', 'City', 'State', 'County', 'Zip', 'Country', 
                      'Continental_indicator', 'REANALYSIS_windspeed_10m_1Day', 'REANALYSIS_lhtfl_DailyMean',
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
                      'Islander_p', 'Other_p', 'Two_p', 'OMAEROe_UVAerosolIndex_Mean', 'NO2_Region',
                      'NLCD_Water100','NLCD_Water10000','PM25_Region','MAIACUS_cosVZA_Aqua_Nearest',
                      'MAIACUS_cosVZA_Terra_Nearest','REANALYSIS_apcp_DailyMean','REANALYSIS_dswrf_DailyMean',
                      'REANALYSIS_gflux_DailyMean','REANALYSIS_shtfl_DailyMean','REANALYSIS_snowc_DailyMean',
                      'REANALYSIS_soilm_DailyMean','REANALYSIS_tcdc_DailyMean','REANALYSIS_vis_DailyMean',
                      'REANALYSIS_windspeed_10m_DailyMin','REANALYSIS_prate_DailyMin','REANALYSIS_vis_DailyMin',
                      'REANALYSIS_prate_1Day','REANALYSIS_vis_1Day','MOD09A1','MOD11A1_Clear_night_cov_Nearest4',
                      'Nearby_Peak2Lag3_Ozone','OMAEROe_UVAerosolIndex_Mean','OMAEROe_VISAerosolIndex_Mean',
                      'OMAERUVd_UVAerosolIndex_Mean','OMO3PR','OMTO3e_ColumnAmountO3','OMUVBd_UVindex_Mean',
                      'NLCD_Herbaceous10000','RoadDensity_primaryroads1000','RoadDensity_prisecroads1000',
                      'Business_Restaurant1000','Ozone_Region','REANALYSIS_hpbl_DailyMax','REANALYSIS_vis_DailyMax',
                      'MOD11A1_LST_Day_1km_Nearest4','MOD13A2_Nearest4','Nearby_Peak2_NO2','Nearby_Peak2Lag1_NO2',
                      'Nearby_Peak2Lag3_NO2','Some_College_p','Native_p','Asian_p','Hispanic_p','Income_50to75k_p',
                      'Age_0to9_p', 'Age_10to19_p', 'Age_20to29_p', 'Age_30to39_p', 'Age_40to49_p', 'Age_over70_p',
                      'Income_0to25k_p', 'Income_25to50k_p', 'Income_75to100k_p', 'Income_over200k_p', 'Family_Household_p',
                      'NLCD_Barren10000', 'NLCD_Wetlands10000', 'OMSO2e_ColumnAmountSO2_PBL_Mean',
                      'High_School_p', 'No_Diploma_p', 'Graduate_Degree_p')

#Remove variables to drop from data
full_data = select(full_data, -one_of(variables_to_drop))

#Convert date to date format
full_data$date = as.Date(full_data$date)

#Extract month from date
full_data$month = as.numeric(format(full_data$date,'%m'))

#Redefine year as number of years from starting point
full_data$year = full_data$year - min(full_data$year)

#Engineer other date features
full_data$cumulative_month = full_data$year*12 + full_data$month 
#full_data$day_of_year = as.numeric(format(full_data$date,'%d')) + 30*(full_data$month-1)
#full_data$cumulative_day =  365*(full_data$year) + full_data$day_of_year

#https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
full_data = full_data %>% mutate(sin_time = round(sin(2*pi*month/12), 8),
                                 cos_time = round(cos(2*pi*month/12), 8))

#Move date variables to the front of the data frame
full_data = full_data %>% select(site, year, month, cumulative_month, sin_time, cos_time,
                                 everything())

#Create latitude, longitude interaction variable 
full_data$Lat_Lon_int = full_data$Lat * full_data$Lon 

#create lead and site mean variables for nearby PM2.5
full_data = full_data %>% arrange(site, date) %>% group_by(site) %>% 
  mutate(Nearby_Peak2_PM25_Lead1 = lead(Nearby_Peak2_PM25, default = mean(Nearby_Peak2_PM25, na.rm = T), order_by = date),
         Nearby_PM25_Site_Mean = mean(Nearby_Peak2_PM25, na.rm = T)) %>%
  ungroup() %>% as.data.frame()

#Remove date variable
full_data = select(full_data, -date)

#Write to csv 
fwrite(full_data, '../data/data_to_impute.csv')

print('Done')
