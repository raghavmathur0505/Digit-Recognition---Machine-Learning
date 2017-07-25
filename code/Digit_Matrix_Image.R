#install packages
install.packages("ROCR")
library(ROCR)
#create a 28x28 matrix for data

digit_matrix = matrix(unlist(digit_train[784,-1]),byrow = T, nrow = 28)
# Plot that matrix

rotate_digit <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)

# Plot a bunch of images
par(mfrow=c(3,4))
lapply(1:9, 
       function(x) image(
         rotate_digit(matrix(unlist(digit_train[x,-1]),byrow = T,nrow = 28)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)

