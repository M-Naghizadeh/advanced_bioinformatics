##Create an example dataframe with 100 rows and 100 columns
df = data.frame(matrix(runif(10000), nrow = 100, ncol = 100, dimnames = list(paste('Row','_', seq(100), sep = ''), paste('Var','_', seq(100), sep = ''))))

#Checkout for descriptive stat for this data
df_descriptive_stat = data.frame(
    'min' = apply(df, 2, min),
    'max' = apply(df, 2, max),
    'mean' = apply(df, 2, mean),
    'var' = apply(df, 2, var),
    'sd' = apply(df, 2, sd),
    'Q1' = apply(df, 2, quantile, 1/4),
    'Q2' = apply(df, 2, quantile, 3/4))

head(df_descriptive_stat)
dim(df_descriptive_stat)

#Check out for the correlation between variables
corr_matrix = round(cor(df), 2)
head(corr_matrix)

#Corrgram
library(corrgram)
corrgram(scale(df), order = T) #its very important to set the order = True in order to gather the correlated vars newa to each other.
dev.off()


#
res_pca = prcomp(df, scale = T, center = T)
names(res_pca)
res_pca$scale 
res_pca$center #the mean of each variable
res_pca$sdev
head(res_pca$rotation) #This is the loading for each variable
head(res_pca$x); #this is the score for ewach sample for each principle component


#Visualize score plot and biplot
plot(res_pca, col = 'lightblue')
plot(res_pca$x[,c('PC2', 'PC3')], col = 'blue', fill = 'red')


#Scoreplot with ggplot
library(ggplot2)
ggplot(data.frame(res_pca$x), aes(PC1,PC2)) +
    geom_point(color = 'blue', 
                size = 2, 
                alpha = .3)+
    stat_ellipse(col = 'red', level = .9, type ='t')+
    labs(x = 'PC1',
         y = 'PC2',
         title = 'Score Plot of principle component 1 and 2')

dev.off()

#plot of variance for each pc
ggplot(data.frame(pca_vars), aes(x = seq(100),y = pca_vars))+
    geom_point()+
    labs(x = 'PC')



library(factoextra)
fviz_eig(res_pca) #Scree plot
dev.off()

fviz_pca_ind(res_pca) #Scoreplot
dev.off()

fviz_pca_biplot(res_pca) #biplot of samples and variables
dev.off()





#screeplot
prop_var <- pca_vars/sum(pca_vars)
prop_var
ggplot(data.frame(prop_var), aes(x= seq(100), y = prop_var))+
    geom_point()+
    labs(x= 'pc')

dev.off()


#cumulative screeplot
ggplot(data.frame(cumsum(prop_var)), aes(x = seq(100), y = cumsum(prop_var)))+
    geom_point()+
    geom_hline(yintercept = .9, col = 'red')

dev.off()



library(devtools)
install_github('vqv/ggbiplot' , dependencies = T)
library('ggbiplot', choice = c(2, 3))
ggbiplot(res_pca, choice = c(2, 4), labels = rownames(df))
dev.off()

groups = c(rep('group1', 30), rep('group2', 40), rep('group3', 30))
ggbiplot(res_pca, choice = c(2, 4), labels = rownames(df), groups = groups, ellipse = T)
dev.off()



#Refrences:
#1.
#2.
#3.
