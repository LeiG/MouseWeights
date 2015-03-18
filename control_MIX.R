###############################################
## Fit mixed effects model for control group ##
###############################################
require(lme4)
require(lattice)
require(boot)

## data manipulation ##
raw<- read.table("mouse_weights_nomiss.txt", header=T)
contrl<- raw[raw[, 3]==99, ]
contrl$id<- as.factor(contrl$id)

## scale days
#stop()
#contrl$days<- contrl$days - 365
contrl$days<- contrl$days/365 - 1

## visulization ##
subgrp<- sample(contrl$id, 18)
contrl_subgrp<- contrl[contrl$id %in% subgrp, ]
# linear fit
# xyplot(weight ~ days | id, data=contrl_subgro, 
#        panel = function(x, y){
#          panel.xyplot(x, y)
#          panel.lmline(x, y)
#        }, ylim=c(20, 60), as.table=T)

xyplot(contrl$weight~contrl$days, type="o", main="Time vs. Weight", xlab="Time in Days", ylab="Weight")

# smooth fit
xyplot(weight ~ days | id, data=contrl_subgrp,
       prepanel = function(x, y) prepanel.loess(x, y, family="gaussian"),
       xlab = "Age", ylab = "Tolerance",
       panel = function(x, y) {
         panel.xyplot(x, y)
         panel.loess(x,y, family="gaussian") },
       ylim=c(30, 60), as.table=T)


## model fitting ##
fm1<- lmer(weight ~ days + (days | id), data=contrl, REML=F)
fm1<- lmer(weight ~ days + I(days^2) + (days + I(days^2) | id), data=contrl, REML=F)
summary(fm1)
# look for getME()
d3<- getME(fm1, "fixef")
d4<- (vcov(fm1))^(-1)
sigma<- getME(fm1, "sigma")
lambda_D<- sd(getME(fm1, "b"))^(-2)
# boostrap estimation
lambda_calc<- function(d, i){
  #sd(d[i, ])^(-2)
  sd(c(ranef(fm1)$id[i,][,1],ranef(fm1)$id[i,][,2]))^(-2)
}
#lambda_boot<- boot(getME(fm1, "b"), lambda_calc, R=500)
lambda_boot<- boot(ranef(fm1)$id, lambda_calc, R=100)
mean(lambda_boot$t)
var(lambda_boot$t)
d2<- mean(lambda_boot$t)/var(lambda_boot$t)
d1<- mean(lambda_boot$t)*d2

# fit uncorrelated random effects
# fm2<- lmer(weight ~ days + (days || id), data=contrl, REML=F)
# summary(fm2)
# getME(fm2, "fixef") # coef for fixed effects
# vcov(fm2) # variancee-covariance for fixed effects
# getME(fm2, "Lambda") # covariance factor of random effects
# sigma(fm2) # residual std
# getME(fm2, "b") # random effects
# ranef(fm2) # random effects
