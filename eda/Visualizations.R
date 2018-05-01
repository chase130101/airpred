require(data.table)
require(dplyr)

#setwd("~/GitHub/airpred")
var_importances = fread('./data/rf_feature_importances.csv')
str(var_importances)

#Variable Importance Plot
var_importances = var_importances %>% 
  mutate(RF_Feature_Importance = RF_Feature_Importance/max(RF_Feature_Importance)) 
sorted_vi = var_importances %>% arrange(desc(RF_Feature_Importance)) %>% slice(1:10)

ggplot(sorted_vi, aes(x = reorder(Variable, RF_Feature_Importance), weight = RF_Feature_Importance)) + 
  geom_bar(fill = 'dodgerblue') + ggtitle('Variable Importances') + 
  ylab('Relative Importance') + xlab('Variable') + coord_flip()


r2_scores = fread('./data/r2_scores_ridgeImp.csv')
str(r2_scores)

missing_r2_scores = filter(r2_scores, Train_num_missing > 0)

ggplot(missing_r2_scores, aes(x = reorder(Variable, -Test_R2), weight = Test_R2)) + 
  geom_bar(fill = 'dodgerblue') + ggtitle('Imputation Performance') + 
  ylab('R-squared') + xlab('') + 
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))



r2_scores = fread('./data/r2_scores_rfImp.csv')
str(r2_scores)

missing_r2_scores = filter(r2_scores, Train1_num_missing > 0 | Train2_num_missing  > 0)

ggplot(missing_r2_scores, aes(x = reorder(Variable, -Test_R2), weight = Test_R2)) + 
  geom_bar(fill = 'dodgerblue') + ggtitle('Imputation Performance') + 
  ylab('R-squared') + xlab('') + 
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))



#setwd("C:/Users/Keyan/Google Drive/Harvard/AC 297R")
full_rf_pred = fread('test_rfPred.csv')
names(full_rf_pred)

rf_pred = full_rf_pred %>% select(c(MonitorData, MonitorData_pred, Lat, Lon, site))

#R-squared
r_sq = function(actual, pred.y){
  
  R2 = 1 - (sum((actual - pred.y)^2) / sum((actual - mean(actual))^2))
  return(R2)
  
}

site_pred_error = rf_pred %>% group_by(site) %>% 
  summarize(mean_error = mean(abs(MonitorData - MonitorData_pred)), 
            r_sq = r_sq(MonitorData, MonitorData_pred),
            Lat = Lat[1], Lon = Lon[1])


ggplot(data = site_pred_error, aes(x = Lon, y = Lat, col = r_sq)) + 
  borders('state', size = 1.25) + 
  geom_point(size = 5) +
  scale_color_gradient2('R-squared', limits=c(0, 1), low = 'white', high = 'red') + 
  ggtitle('Model Performance by Location') 


ggplot(data = site_pred_error, aes(x = Lon, y = Lat)) + 
  borders('state', size = 1.25) + 
  geom_point(size = 5, col = 'yellow') +
  geom_point(data = location_census_data, aes(Lon, Lat), size = 2, col = 'black')


#setwd("~/GitHub/airpred")
location_census_data = fread('./data/sensor_locations_with_census.csv')
graph_data = location_census_data %>% filter(Continental_indicator == 1) %>%
             mutate(Data = ifelse(site %in% rf_pred$site, 'Test', 'Train'))
graph_data = left_join(graph_data, site_pred_error)
str(graph_data)

ggplot(data = graph_data, aes(x = Lon, y = Lat, col = Data)) + 
  borders('state', size = 1.25) + 
  geom_point(size = 3) +
  ggtitle('Sensor Test-Train Split')
