# Master R in one Day 
# Xie Yu Ting
# 2022.6.25

setwd("/home/tt/Codes/ML/basic/R")
options(scipen = 100)

# Load data from CSV into data frame
df <- read.csv("./data_in/control.csv", header=TRUE, sep = ",",  stringsAsFactors = FALSE)

# Write data to CSV file
lat <- data.frame(df$"component", df$"ts_start", df$"ts_end" - df$"ts_cam", df$"ts_end" - df$"ts_lidar", df$"ts_end" - df$"ts_radar", df$"is_finish")
# write.csv(lat[1:10,], file = "./data_out/latency_data.csv", append = FALSE, quote = TRUE, sep = ",")

# Update column names
colnames(lat) <- c("component", "ts_start", "lat_cam", "lat_lidar", "lat_radar", "is_finish")

# print(lat[1:10, ])

lat$"lat_cam" = lat$"lat_cam" %/% 1e6
lat$"lat_lidar" = lat$"lat_lidar" %/% 1e6
lat$"lat_radar" = lat$"lat_radar" %/% 1e6

# Mapping operation element-wise
lat[lat$"lat_cam" > 1000, ]$"lat_cam" <- 0
lat[lat$"lat_lidar" > 1000, ]$"lat_lidar" <- 0
lat[lat$"lat_radar" > 1000, ]$"lat_radar" <- 0

# For iteration
rows <- nrow(lat)
for (idx in 1: rows) {
    if (lat[idx, "is_finish"] != 0) {
        break
    }
}

# Slice data frame
lat <- lat[idx: rows, ]
# print(lat[1: 10, c("component", "ts_start", "lat_cam", "lat_lidar", "lat_radar")])

# Basic stats
print(sprintf("Mean: %.2f", mean(lat$"lat_cam")))
print(sprintf("Var: %.2f", var(lat$"lat_cam")))
print(sprintf("SD: %.2f", sd(lat$"lat_cam")))

# Basic plots
plot(1: nrow(lat), lat$"lat_cam", type="l")
boxplot(lat$"lat_cam")

print("Done!")