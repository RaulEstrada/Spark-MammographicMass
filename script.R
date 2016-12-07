# Load CSV data file to compute and visualize data correlation
data <- read.csv("C:/Users/RaulEstrada/Downloads/mammographic/mammographic.dat")
completeData <- data[complete.cases(data),]
dataCorr <- cor(completeData)

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

# Analyze balance of Severity population
severe <- sum(completeData$Severity==1)
notSevere <- sum(completeData$Severity==0)
print(paste("Severe:", severe, ". Not Severe:", notSevere))


# Divide data set into training data (70%) and test data (30%)
ind <- sample(2, nrow(completeData), replace = TRUE, prob = c(.7, .3))
trainData <- completeData[ind==1,]
testData <- completeData[ind==2,]

# Create and train the random forest
library(randomForest)
randomForestGen <- randomForest(Severity ~., data = trainData, ntree = 100, proximity = TRUE)

# Plot random forest generated
plot(randomForestGen)

# Display importance of variables and display it
importance(randomForestGen)
varImpPlot(randomForestGen)
