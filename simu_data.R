#-----------------------------------------------------------------------------
# Simulate a toy data set under the settings of the proposed BVS model.
#-----------------------------------------------------------------------------
require(MASS)
require(reshape2)
require(lme4)
require(lattice)

set.seed(3)

#### parameter settings ####
days<- c(365,395,456,517,578,639,700,760,821,882,943,1004,1065,1125,1186)
W<- matrix(0, nrow=length(days), ncol=2)
W[,1]<- 1
W[,2]<- days/365 - 1 # standardize
Z<- W
X<- matrix(W[,2], ncol=1)
idmatrix<- diag(length(days))

#### prior settings ####
d1 = 75.95
d2 = 871.47

# small random effects
# d1 = 6.85
# d2 = 5.85

## estimated from variance of the Random effects for 'Intercept' and 'days'
# hist(rgamma(1000, d1, d2), 100)
d3 = matrix(c(45.50, -5.75), ncol=1)
d4_inv = matrix(c(0.04, -0.02, -0.02, 0.06), nrow=2, ncol=2)

#### initial values ####
sigma2<- 5.06
lambdaD<- 1/((9.065+14.261)/2)

# small random effects
# lambdaD<- 1/1.0

alpha<- matrix(c(45.50, -5.75), ncol=1)

#### generate control grp ####
n_contrl<- 2266
y_contrl<- matrix(0, nrow=n_contrl, ncol=length(days))
for(i in 1:n_contrl){
  y_contrl[i,]<- W%*%alpha +
                 Z%*%matrix(mvrnorm(1, matrix(0, nrow=2, ncol=1), diag(2)/rgamma(1, d1, d2)), ncol=1) +
                 matrix(mvrnorm(1, matrix(0, nrow=length(days), ncol=1), sigma2*idmatrix), ncol=1)
}

colnames(y_contrl)<- days

contrl_grp<- as.data.frame(y_contrl)
contrl_grp$id<- as.factor(seq(1, n_contrl))
contrl_grp$diet<- as.factor(99)

## convert to long format
contrl_grp<- melt(contrl_grp, id.vars = c('id', 'diet'), variable.name  = 'days', value.name = 'weight')
contrl_grp<- contrl_grp[order(contrl_grp$id, contrl_grp$days), ]
contrl_grp$days<- as.numeric(as.character(contrl_grp$days))
contrl_grp = contrl_grp[, c("days", "id", 'diet', 'weight')]

## save to file
write.table(contrl_grp, "simu_control_grp.txt", quote = FALSE, row.names=FALSE)

## visualize
plot(days, apply(y_contrl, 2, mean), 'o', main="Time vs. Mean Weight", xlab="Time", ylab="Mean Weight")
xyplot(contrl_grp$weight~contrl_grp$days, type="o", main="Time vs. Weight", xlab="Time in Days", ylab="Weight")
# xyplot(weight ~ days | id, data=contrl_grp,
#        prepanel = function(x, y) prepanel.loess(x, y, family="gaussian"),
#        xlab = "Age", ylab = "Tolerance",
#        panel = function(x, y) {
#          panel.xyplot(x, y)
#          panel.loess(x,y, family="gaussian") },
#        ylim=c(30, 60), as.table=T)

## mixed effects model1
contrl_model<- contrl_grp
contrl_model$days<- contrl_model$days/365-1
fm_contrl<- lmer(weight ~ days + (days | id), data=contrl_model, REML=F)
summary(fm_contrl)

#### generate treatment grp ####
## add Beta params to modify Alpha
beta<- -2

## simulate data
n_trt<- 1000
y_trt<- matrix(0, nrow=n_trt, ncol=length(days))
for(i in 1:n_trt){
  y_trt[i,]<- W%*%alpha + X*beta +
    Z%*%matrix(mvrnorm(1, matrix(0, nrow=2, ncol=1), diag(2)/rgamma(1, d1, d2)), ncol=1) +
    matrix(mvrnorm(1, matrix(0, nrow=length(days), ncol=1), sigma2*idmatrix), ncol=1)
}

colnames(y_trt)<- days

trt_grp<- as.data.frame(y_trt)
trt_grp$id<- as.factor(seq(n_contrl+1, n_contrl+n_trt))
trt_grp$diet<- as.factor(1)

## convert to long format
trt_grp<- melt(trt_grp, id.vars = c('id', 'diet'), variable.name  = 'days', value.name = 'weight')
trt_grp<- trt_grp[order(trt_grp$id, trt_grp$days), ]
trt_grp$days<- as.numeric(as.character(trt_grp$days))
trt_grp = trt_grp[, c("days", "id", 'diet', 'weight')]

## save to file
simu_grp<- rbind(trt_grp, contrl_grp)
write.table(simu_grp, "simu_grp_3.txt", quote = FALSE, row.names=FALSE)

## visualize
lines(days, apply(y_trt, 2, mean) , 'o', main="Time vs. Mean Weight", xlab="Time", ylab="Mean Weight")
xyplot(trt_grp$weight~trt_grp$days, type="o", main="Time vs. Weight", xlab="Time in Days", ylab="Weight")
# xyplot(weight ~ days | id, data=trt_grp,
#        prepanel = function(x, y) prepanel.loess(x, y, family="gaussian"),
#        xlab = "Age", ylab = "Tolerance",
#        panel = function(x, y) {
#          panel.xyplot(x, y)
#          panel.loess(x,y, family="gaussian") },
#        ylim=c(30, 60), as.table=T)

## mixed effects model1
fm_trt<- lmer(weight ~ days + (days | id), data=trt_grp, REML=F)
summary(fm_trt)





