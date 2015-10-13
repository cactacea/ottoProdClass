library(nnet)
#library(randomForest)
library("gbm")
#library(neuralnet)

projectLoc = "D:/ml/projects/productClass/"
#projectLoc = "C:/Users/kpasad/data/ML/projects/productClass/"
trainFile = "train.csv"
testFile ="test.csv"
submissionFile = "submission.csv"
resultsFile="results.cvs"

cvFactor =0.7
cv_folds =5
epsilon = 1e-15
trainData = read.csv(paste0(projectLoc,trainFile),header=TRUE)
testData = read.csv(paste0(projectLoc,testFile),header=TRUE)
resultsFile=paste0(projectLoc,"results.csv")
numSamples = dim(trainData)[1]
numTrainSamplesPerCV=round(numSamples*cvFactor)
numTestSamplesPerCV = numSamples-numTrainSamplesPerCV

multinom_maxit=300

feats<-colnames(trainData)
feats<-feats[-1]
feats<-feats[-length(feats)]


model = 'gbm'
rf_ntrees=500
gbm_distribution = "multinomial"
gbm_n_tree=2
gbm_shrinkage=0.1
gbm_verbose=TRUE
gbm_interaction_depth=2
gbm_bag_fraction=0.9
gbm_predict_n_trees=gbm_n_tree
nnet_size = 120 
nnet_maxit=25
nnet_MaxNWts=50000
nnet_linout=TRUE 


#gbm_n_tree<-c(10,50,100,200,500,700,1000)
#gbm_shinkage<-c(0.5,0.1,0.05,0.01)
gbm_interaction_depth<-c(2,4,6,8,10)
gbm_distribution='multinomial'
gbm_params<-data.frame(gbm_distribution,gbm_n_tree,gbm_shrinkage,gbm_interaction_depth,gbm_bag_fraction)

for (cv_cnt in 1:cv_folds){
  trainSampleIdx = sample(1:numSamples,numTrainSamplesPerCV,replace=FALSE)
  testSampleIdx = c(1:numSamples)[-c(trainSampleIdx)]
  
  if (model == 'multinom') {    
    model_multinom <- multinom(trainData$target[trainSampleIdx] ~ ., data = trainData[trainSampleIdx,-c(1,ncol(trainData))],maxit=multinom_maxit)
    predictions<-predict(model_multinom,trainData[testSampleIdx,-c(1,ncol(trainData))] , type='probs')
    }
  
  if (model == 'RF') {    
    model_rf <- randomForest(trainData$target[trainSampleIdx] ~ ., data=trainData[trainSampleIdx,-c(1,ncol(trainData))],ntree=rf_ntrees)
    predict_train<-predict(model_rf,newdata=(trainData[trainSampleIdx,-c(1,ncol(trainData))]),type="prob")
    predict_test <- predict(model_rf,newdata=(trainData[testSampleIdx,-c(1,ncol(trainData))]),type="prob")
  }
  
  lables=trainData$target[trainSampleIdx] 
  trainData=trainData[trainSampleIdx,-c(1,ncol(trainData))]
  if (model == 'gbm') {
    #apply(gbm_params,1,gbm_sweep(gbm_params,lables,trainData),lables,trainData)
    predict_train<-apply(gbm_params,1,function(gbm_params,lables,trainData){
      
      gbm_distribution<-gbm_params['gbm_distribution']
      gbm_n_tree<-gbm_params['gbm_n_tree']
      gbm_shrinkage<-gbm_params['gbm_shrinkage']
      gbm_interaction_depth<-as.numeric(gbm_params['gbm_interaction_depth'])
      gbm_bag_fraction<-gbm_params['gbm_bag_fraction']
      
      
        model_gbm <-gbm(lables ~ ., trainData,distribution = gbm_distribution,n.tree=gbm_n_tree,shrinkage=gbm_shrinkage,interaction.depth=gbm_interaction_depth,verbose=TRUE)
        predict_test <- predict(model_gbm,newdata=trainData[testSampleIdx,-c(1,ncol(trainData))],type="response",n.trees=gbm_predict_n_trees)
        predict_train<-predict(model_gbm,newdata=trainData[trainSampleIdx,-c(1,ncol(trainData))],type="response",n.trees=gbm_predict_n_trees)          
    },lables,trainData)
    #gbm_sweep(gbm_distribution,gbm_n_tree,gbm_shrinkage,gbm_verbose,gbm_interaction_depth,gbm_bag_fraction,lables,trainData)    
  }
  if (model == 'nnet') {    
    model_nnet<-nnet(trainData$target[trainSampleIdx]~., data=scale((trainData[trainSampleIdx,-c(1,ncol(trainData))])+1),size = nnet_size, maxit=nnet_maxit,MaxNWts=nnet_MaxNWts,linout=nnet_linout )  
    predict_train<-predict(model_nnet,newdata=scale(log(trainData[trainSampleIdx,-c(1,ncol(trainData))]+1)),type="raw")
    predict_test <-predict(model_nnet,newdata=scale(log(trainData[ testSampleIdx,-c(1,ncol(trainData))]+1)),type="raw")
  }
  
  y_true <-class.ind(trainData$target[testSampleIdx])
  y_true_train<-class.ind(trainData$target[trainSampleIdx])
  log_loss_train = -sum(y_true_train*log(pmin(1-epsilon,pmax(predict_train,epsilon))))/dim(y_true_train)[1]
  log_loss_test = -sum(y_true*log(pmin(1-epsilon,pmax(predict_test,epsilon))))/dim(y_true)[1]
  
  params_df = data.frame(model,log_loss_train,log_loss_test,cv_folds,cv_cnt,
               multinom_maxit,
               gbm_distribution,gbm_n_tree,gbm_shrinkage,gbm_verbose,gbm_interaction_depth,gbm_bag_fraction,gbm_predict_n_trees,
               nnet_size,nnet_maxit,nnet_MaxNWts,nnet_linout) 
  write.csv(params_df,resultsFile)
  
  cat("cv_fold = ",cv_cnt, "log_loss_train = ",log_loss_train,"log_loss_test = ",log_loss_test)
}


#test_preds<-predict(model,testData[,-1],type='probs')
#prwrite.csv(cbind(testData$id,test_preds),paste0(projectLoc,submissionFile))

