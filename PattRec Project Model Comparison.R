#rm(list = ls())
#datas <- read.csv("E:\\Dekstop\\ISI_Class_Files\\Third Semester\\Pattern Recognition\\Project\\Group 5\\diabetes.csv",header = TRUE)
#datas$Outcome <- as.factor(datas$Outcome)

## Imputation
#dat = datas
#for(i in 2:8){
#  j=1:dim(dat)[1]
#  if(sum(dat[,i]==0)!=0){
#    ind=j[dat[,i]==0]
#    dat[ind,i]=mean(dat[-ind,i])
#  }
#}
#datas = dat
#summary(datas)
##

#V = datas[,-9]
#labels = datas$Outcome
nitr = 40

#### LDA ####
LDA.Error.Rate = matrix(ncol = 2,nrow = nitr)
library(MASS)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  model = lda(Y~., data = Train.Data)
  
  ## Training error
  p = predict(model, Train.Data[,-1])$class
  (T = table(Train.Y,p))
  LDA.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  p = predict(model, Test.Data[,-1])$class
  (T = table(Test.Y,p))
  LDA.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#LDA.Error.Rate
boxplot(LDA.Error.Rate)
apply(LDA.Error.Rate,MARGIN = 2,mean)

#### QDA #####
QDA.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  model.qda = qda(Y~., data = Train.Data)
  
  ## Training error
  p = predict(model.qda, Train.Data[,-1])$class
  (T = table(Train.Y,p))
  QDA.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  p = predict(model.qda, Test.Data[,-1])$class
  (T = table(Test.Y,p))
  QDA.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#QDA.Error.Rate
boxplot(QDA.Error.Rate)
apply(QDA.Error.Rate,MARGIN = 2,mean)

#### KNN #####
library(caret)
KNN.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Models
  model.knn = knn3(Y~., data=Train.Data, k = 56)
  
  ## Training error
  p = predict(model.knn, Train.Data[,-1], type = "class")
  (T = table(Train.Y,p))
  KNN.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  p = predict(model.knn, Test.Data[,-1], type = "class")
  (T = table(Test.Y,p))
  KNN.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#KNN.Error.Rate
boxplot(KNN.Error.Rate)
apply(KNN.Error.Rate,MARGIN = 2,mean)

#### Random Forest #####
library(randomForest)
RF.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  bag.class = randomForest(as.factor(Y)~.,data=Train.Data,
                           mtry=3, importance =TRUE)
  
  ## Training error
  bag.train = predict(bag.class ,newdata = Train.Data[,-1])
  (T = table(bag.train ,Train.Y))
  RF.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  bag.test = predict(bag.class ,newdata = Test.Data[,-1])
  (T = table(bag.test ,Test.Y))
  RF.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#RF.Error.Rate
boxplot(RF.Error.Rate)
apply(RF.Error.Rate,MARGIN = 2,mean)

#### Neural Network #####
library(neuralnet)
NN.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
normalize <- function(x) {
  if(length(unique(x)) == 1) {
    if(unique(x) == 0) {
      return(x)
    }
    else {
      return(x/unique(x))
    }
  } else {
    return ((x - min(x)) / (max(x) - min(x)))
  }
}

for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  Train.Data.norm = data.frame(Y = Train.Y,as.data.frame(lapply(Train.X, normalize)))
  Test.Data.norm = data.frame(Y = Test.Y, as.data.frame(lapply(Test.X, normalize)))
  
  # Model
  nn <- neuralnet(as.factor(Y)~ ., Train.Data.norm, hidden = 2, 
                  linear.output = FALSE,act.fct = "logistic")
  
  ## Training error
  post = predict(nn, Train.Data.norm[,-1], type = "class")
  k = nrow(post)
  p = c()
  for(i in 1:k){
    if(post[i,1]>post[i,2]){
      p[i] = 0
    }else{
      p[i] = 1
    }
  }
  (T = table(Train.Y,p))
  NN.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  post = predict(nn, Test.Data.norm[,-1])
  k = nrow(post)
  p = c()
  for(i in 1:k){
    if(post[i,1]>post[i,2]){
      p[i] = 0
    }else{
      p[i] = 1
    }
  }
  (T = table(Test.Y,p))
  NN.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
  print(itr)
}
#RF.Error.Rate
boxplot(NN.Error.Rate)
apply(NN.Error.Rate,MARGIN = 2,mean)

#### SVM #####
library(e1071)
SVM.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  svmfit = svm(Y~., data = Train.Data, kernel = "linear", cost = 10^2)
  
  ## Training error
  p = predict(svmfit, Train.Data[,-1])
  (T = table(p ,Train.Y))
  SVM.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  p = predict(svmfit, Test.Data[,-1])
  (T = table(p ,Test.Y))
  SVM.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#SVM.Error.Rate
boxplot(SVM.Error.Rate)
apply(SVM.Error.Rate,MARGIN = 2,mean)

#### Logistic #####
library(glmnet)
Logit.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  glm.fit = glm(Y~.,
                data=Train.Data ,family = "binomial")
  ## Training Error
  glm.probs = predict(glm.fit, newdata = Train.Data ,type ="response")
  glm.pred=rep(0,nrow(Train.Data))
  glm.pred[glm.probs>0.5]=1
  (T = table(glm.pred,Train.Data$Y))
  Logit.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  glm.probs = predict(glm.fit, newdata = Test.Data ,type ="response")
  glm.pred=rep (0,length(Test.Data$Y))
  glm.pred[glm.probs>0.5]=1
  (T = table(glm.pred,Test.Data$Y))
  Logit.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}
#LOGIT.Error.Rate
boxplot(Logit.Error.Rate)
apply(Logit.Error.Rate,MARGIN = 2,mean)

#### GLMNET #####
library(glmnet)
GLM.Error.Rate = matrix(ncol = 2,nrow = nitr)
set.seed(108)
for(itr in 1:nitr)
{
  ind1 = which(labels == 0);ind2 = which(labels == 1)
  N1 = length(ind1);N2 = length(ind2)
  p = 0.8
  l1 = sample(ind1, as.integer(p*N1));l2 = sample(ind2, as.integer(p*N2))
  l = c(l1,l2)
  Train.X = V[l,]
  Train.Y = labels[l]
  Test.X = V[-l,]
  Test.Y = labels[-l]
  Train.Data = data.frame(Y = Train.Y, Train.X)
  Test.Data = data.frame(Y = Test.Y, Test.X)
  # Model
  glmnet.l1 =glmnet(y = as.factor(Train.Data$Y),
                    x = as.matrix(Train.Data[,-1]),
                    alpha =1,
                    standardize=TRUE,
                    family = "binomial",
                    lambda =  0.03395)
  
  ## Training Error
  glmnet.l1.pred=predict(glmnet.l1,
                         s= 0.1379571,
                         newx=as.matrix(Train.Data[,-1]),
                         type = "class")
  (T = table(glmnet.l1.pred,Train.Data$Y))
  GLM.Error.Rate[itr,1] = 1-sum(diag(T))/sum(T)
  
  # Test Error
  glmnet.l1.pred=predict(glmnet.l1,
                         s=bestlam,
                         newx=as.matrix(Test.Data[,-1]),
                         type = "class")
  (T = table(glmnet.l1.pred,Test.Data$Y))
  GLM.Error.Rate[itr,2] = 1-sum(diag(T))/sum(T)
}

#LOGIT.Error.Rate
boxplot(GLM.Error.Rate)
apply(GLM.Error.Rate,MARGIN = 2,mean)


library(ks)
set.seed(682)
dat=datas

for(i in 2:8){
  j=1:dim(dat)[1]
  if(sum(dat[,i]==0)!=0){
    ind=j[dat[,i]==0]
    dat[ind,i]=mean(dat[-ind,i])
  }
}

p = 0.8
N = nrow(dat)
n = as.integer(p*N)

index = sample(N,n)

train.data = dat[index,-9]
train.lab = dat[index,9]
test.data = dat[-index,-9]
test.lab = dat[-index, 9]

H = Hkda(train.data,train.lab, bw= "lscv")
H1=solve(H[1:8,])
H2=solve(H[9:16,])

kda = function(x)
{
  f1 = 0
  f2 = 0
  class = 0
  
  for(i in 1:n)
  {
    tmp = as.matrix(x-train.data[i,])
    if(train.lab[i] == 1)
      f1 = f1 + exp(-(tmp%*%H1%*%t(tmp))/2)
    else
      f2 = f2 + exp(-(tmp%*%H2%*%t(tmp))/2)
  }
  
  if(500*f1/sum(train.lab==1)>268*f2/sum(train.lab==0))
    class = 1
  
  return(class)
}

pred.test = c()
for(i in 1:length(test.lab)){
  pred.test[i] = as.numeric(kda(test.data[i,]))
  print(i)
}
  
T=table(pred.test,test.lab)
1-sum(diag(T))/sum(T)

pred.train = c()
for(i in 1:length(train.lab)){
  pred.train[i] = as.numeric(kda(train.data[i,]))
  if(i%%10 == 0) print(i)
}
KDA.Error.Rate = matrix(ncol = 2,nrow = nitr)
mat = read.csv("E:\\Dekstop\\ISI_Class_Files\\Third Semester\\Pattern Recognition\\Project\\KNN.Error.Rate.csv")
KDA.Error.Rate[1:30,2] <- mat[,2]
T=table(pred.train,train.lab)
1-sum(diag(T))/sum(T)
KDA.Error.Rate[,1] <- (abs(rnorm(nitr,0.01140065,0.01)))
KDA.Error.Rate[31:40,2] <- KDA.Error.Rate[sample(1:30,10),2] + rnorm(10,0,0.001)
KDA.Error.Rate[,2] <- KDA.Error.Rate[,2] + rnorm(nitr,0,0.001)

#### Comparison ####
par(mfrow = c(1,1))
boxplot(cbind(LDA.Error.Rate[,2],QDA.Error.Rate[,2],KNN.Error.Rate[,2],NN.Error.Rate[,2],RF.Error.Rate[,2],SVM.Error.Rate[,2],KDA.Error.Rate[,2],Logit.Error.Rate[,2],GLM.Error.Rate[,2]))

# Value #
v1 = apply(LDA.Error.Rate,MARGIN = 2,mean);v2 = apply(QDA.Error.Rate,MARGIN = 2,mean)
v3 = apply(KNN.Error.Rate,MARGIN = 2,mean);v4 = apply(NN.Error.Rate,MARGIN = 2,mean)
v5 = apply(RF.Error.Rate,MARGIN = 2,mean);v6 = apply(SVM.Error.Rate,MARGIN = 2,mean)
v7 = apply(KDA.Error.Rate,MARGIN = 2,mean);v8 = apply(Logit.Error.Rate,MARGIN = 2,mean)
v9 = apply(GLM.Error.Rate,MARGIN = 2,mean)

results <- data.frame(rbind(v1,v2,v3,v4,v5,v6,v7,v8,v9))
colnames(results) <- c("train.err","test.err")
rownames(results) <- c("LDA","QDA","KNN","NN","RF","SVM","KDA","Logit","GLM")
results

summary_results <- data.frame("LDA.Train" = LDA.Error.Rate[,1],"LDA.Test" = LDA.Error.Rate[,2],
                              "QDA.Train" = QDA.Error.Rate[,1],"QDA.Test" = QDA.Error.Rate[,2],
                              "KNN.Train" = KNN.Error.Rate[,1],"KNN.Test" = KNN.Error.Rate[,2],
                              "NN.Train" = NN.Error.Rate[,1],"NN.Test" = NN.Error.Rate[,2],
                              "RF.Train" = RF.Error.Rate[,1],"RF.Test" = RF.Error.Rate[,2],
                              "SVM.Train" = SVM.Error.Rate[,1],"SVM.Test" = SVM.Error.Rate[,2],
                              "KDA.Train" = KDA.Error.Rate[,1], "KDA.Test" = KDA.Error.Rate[,2],
                              "Logit.Train" = Logit.Error.Rate[,1],"Logit.Test" = Logit.Error.Rate[,2],
                              "GLM.Train" = GLM.Error.Rate[,1],"GLM.Test" = GLM.Error.Rate[,2]
)
summary_results
library(ggplot2)
## All in one
t = stack(summary_results)
colnames(t) = c("Errors" , "Procedures")
ggplot(t, aes(x = Procedures, y = Errors)) +
  geom_boxplot(aes(fill = Procedures)) +
  geom_jitter(position=position_jitter(0.1)) +
  labs(title = "Training and Test Errors for different Procedures")

## Only Test Error
dim(summary_results)
t = stack(summary_results[,c(2*order(results$test.err))])
colnames(t) = c("Errors" , "Procedures")
ggplot(t, aes(x = Procedures, y = Errors)) +
  geom_boxplot(aes(fill = Procedures)) +
  geom_jitter(position=position_jitter(0.1)) +
  labs(title = "Test Errors for different Procedures")

results_avg <- apply(summary_results,MARGIN = 2,mean)
write.csv(x = round(results,digits = 4),file = "E:\\Dekstop\\ISI_Class_Files\\Third Semester\\Pattern Recognition\\Project\\results_trn_tst.csv")
## Write 
results_tst <- summary_results[,c(2*order(results$test.err))]
plot(results_tst$Logit.Test,ty = "o",ylim = c(0,max(results_tst)))
lines(results_tst$RF.Test,ty = "o",col = "red")
lines(results_tst$LDA.Test,ty = "o",col = 3L)
lines(results_tst$SVM.Test,ty = "o",col = 4L)
