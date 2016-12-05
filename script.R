# Load CSV data file to compute and visualize data correlation
data <- read.csv("C:/Users/RaulEstrada/Downloads/mammographic/mammographic.dat")
dataCorr <- cor(data[complete.cases(data),])

# Correlation visualization
library(corrplot)
jpeg(filename = "./dataCorrelation.jpg", quality = 100, pointsize = 20)
corrplot(dataCorr, method = "circle", 
         title = "Mammographic Mass data correlation",
         mar=c(0,0,1,0))
dev.off()

# Store correlation as csv file
write.csv(x = dataCorr, file = "./dataCorrelation.csv", 
          fileEncoding = "utf-8")