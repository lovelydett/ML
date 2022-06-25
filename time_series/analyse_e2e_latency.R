data <- read.csv('./data_in/2022-06-25 15:50:48.617876756_32683.csv', header = TRUE,  sep = ',',  stringsAsFactors = FALSE)

ts <- data.frame(data$'ts_lidar', data$'ts_cam', data$'ts_radar')
lat <- data.frame(data$'ts_lidar' - data$'ts_end', data$'ts_cam' - data$'ts_end', data$'ts_radar'- data$'ts_end')



print(lat[:,])