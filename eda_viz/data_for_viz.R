#Load packages
require(data.table) #used for fread and fwrite
require(dplyr)

data = fread('../data/data_to_impute.csv')
sensor_data = data %>% select(c(site, MonitorData, month, year))
fwrite(sensor_data, '../data/sensor_data.csv')