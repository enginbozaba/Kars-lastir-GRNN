#https://cran.r-project.org/web/packages/grnn/grnn.pdf
library(grnn)

x_1 <- c(0,0,1)
x_2 <- c(1,0,1)
x <- data.frame(x_1,x_2)
y <- c(1,0,0)

veri<-data.frame(y,x)

> veri
  y x_1 x_2
1 1   0   1
2 0   0   0
3 0   1   1


grnn <- learn(veri,variable.column=1)
grnn <- smooth(grnn, sigma=1)

test <- matrix(c(1,0),nrow=1)
colnames(test) <- c('x_1', 'x_2')
> test
    x_1 x_2
[1,]   1   0

guess(grnn, test)


#çıktı : 0.2326965
