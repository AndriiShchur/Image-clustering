library(keras)
if (!require(magick)) install.packages('magick')
library(magick)
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(imager)) install.packages('imager')
library(imager)
if (!require(jpeg)) install.packages('jpeg')
library(jpeg)
if (!require(data.table)) install.packages('data.table')
library(data.table)
if (!require(Rtsne)) install.packages('Rtsne')
library(Rtsne)
if (!require(umap)) install.packages('umap')
library(umap)

#elbow criterion function
wssplot <- function(data, nc=50, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}


# select and  pre-train model 
# model <- application_vgg16(weights = "imagenet", 
#                            include_top = FALSE)

# model <- application_vgg19(weights = "imagenet",
#                            include_top = FALSE)

# model <- application_inception_v3(weights = "imagenet", 
#                            include_top = FALSE)



base_model <- application_vgg19(weights = 'imagenet')
model <- keras_model(inputs = base_model$input, 
                     outputs = get_layer(base_model, 'block4_pool')$output)

# function for images preparation
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224, 224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}


# load data from deep fashion dataset
image_files_path <- ""
#sample size
sample_fashion <- sample(list.dirs(image_files_path), 50)

file_list <- list.files(sample_fashion, full.names = TRUE, recursive = TRUE)

# Get features from models
# VGG-16
vgg16_feature_list <- data.frame()
for (image in file_list) {
  
  print(image)
  cat("Image", which(file_list == image), "from", length(file_list))
  
  vgg16_feature <- predict(model, image_prep(image))
  
  flatten <- as.data.frame.table(vgg16_feature, responseName = "value") %>%
    select(value)
  rgb<-transpose(as.data.table(dim(readJPEG(image))))
  colnames(rgb)<-c('r','g','b')
  flatten <- cbind(image, as.data.frame(t(flatten)),rgb)
  vgg16_feature_list <- rbind(vgg16_feature_list, flatten)
}

# VGG-19
vgg19_feature_list <- data.frame()
for (image in file_list[2166:2795]) {
  
  print(image)
  cat("Image", which(file_list == image), "from", length(file_list))
  
  vgg19_feature <- predict(model, image_prep(image))
  
  flatten <- as.data.frame.table(vgg19_feature, responseName = "value") %>%
    select(value)
  rgb<-transpose(as.data.table(dim(readJPEG(image))))
  if (ncol(rgb)==2) {
    rgb<-cbind(rgb,data.frame(V3=0))
  }

  colnames(rgb)<-c('r','g','b')
  flatten <- cbind(image, as.data.frame(t(flatten)),rgb)
  vgg19_feature_list <- rbind(vgg19_feature_list, flatten)
}

# inception-v3
inception_v3_feature_list <- data.frame()
for (image in file_list) {
  
  print(image)
  cat("Image", which(file_list == image), "from", length(file_list))
  
  inception_v3_feature <- predict(model, image_prep(image))
  
  flatten <- as.data.frame.table(inception_v3_feature, responseName = "value") %>%
    select(value)
  rgb<-transpose(as.data.table(dim(readJPEG(image))))
  colnames(rgb)<-c('r','g','b')
  flatten <- cbind(image, as.data.frame(t(flatten)),rgb)
  inception_v3_feature_list <- rbind(inception_v3_feature_list, flatten)
}


# dimension reduction

# PCA
pca <- prcomp(vgg19_feature_list[, -1],
              center = TRUE,
              scale = FALSE)

#TSNE
tsne <- Rtsne(as.matrix(vgg19_feature_list[, -1]), check_duplicates=FALSE, pca=TRUE, perplexity=30, theta=0.5, dims=3)

#UMAP
umap = umap(as.matrix(vgg19_feature_list[, -1]))

# Calculate cluster number with elbow function and create clustering for all factors

set.seed(50)
n_clust_pca=30
n_clust_tsne=17
n_clust_umap=30

cluster_pca <- kmeans(pca$x, n_clust_pca)
cluster_tsne <- kmeans(cbind(as.data.frame(tsne$Y),as.data.frame(tsne$costs)) , n_clust_tsne)
cluster_umap <- kmeans(umap$layout, n_clust_umap)


cluster_feature_pca <- kmeans(vgg19_feature_list[, -1], n_clust_pca)
cluster_feature_tsne <- kmeans(vgg19_feature_list[, -1], n_clust_tsne)
cluster_feature_umap <- kmeans(vgg19_feature_list[, -1], n_clust_umap)


# Data preparation for vizualization
cluster_list_pca <- data.frame(cluster_pca = cluster_pca$cluster,
                           cluster_feature = cluster_feature_pca$cluster,
                           vgg19_feature_list) %>%
  select(cluster_pca, cluster_feature, image) %>%
  mutate(class = gsub("C:/Users/anshch/Documents/Fashion/Data/img/", "", image),
         class = substr(class, start = 1, stop = 20))

cluster_list_tSNE <- data.frame(cluster_tsne = cluster_tsne$cluster, 
                           cluster_feature = cluster_feature_tsne$cluster,
                           vgg19_feature_list) %>%
  select(cluster_tsne, cluster_feature, image) %>%
  mutate(class = gsub("C:/Users/anshch/Documents/Fashion/Data/img/", "", image),
         class = substr(class, start = 1, stop = 20))

cluster_list_umap <- data.frame(cluster_umap = cluster_umap$cluster, 
                                cluster_feature = cluster_feature_umap$cluster,
                                vgg19_feature_list) %>%
  select(cluster_umap, cluster_feature, image) %>%
  mutate(class = gsub("C:/Users/anshch/Documents/Fashion/Data/img/", "", image),
         class = substr(class, start = 1, stop = 20))



# Plot results

plot_random_images_pca <- function(n_img = 9,
                               cluster = 1,
                               rows = 3,
                               cols = 3) {
  cluster_list_random_cl_1 <- cluster_list_pca %>%
    #cluster_feature cluster_pca
    filter(cluster_feature == cluster) %>%
    sample_n(n_img,replace = TRUE)
  
  graphics::layout(matrix(c(1:n_img), rows, cols, byrow = TRUE))
  for (i in 1:n_img) {
    path <- as.character(cluster_list_random_cl_1$image[i])
    img <- load.image(path)
    plot(img, axes = FALSE)
    title(main = paste("Cluster PCA", cluster))
  }
}

# sapply(c(1), function(x) plot_random_images(cluster = 29))

plot_random_images_tsne <- function(n_img = 9,
                               cluster = 1,
                               rows = 3,
                               cols = 3) {
  cluster_list_random_cl_1 <- cluster_list_tSNE %>%
    #cluster_feature cluster_pca
    filter(cluster_feature == cluster) %>%
    sample_n(n_img,replace = TRUE)
  
  graphics::layout(matrix(c(1:n_img), rows, cols, byrow = TRUE))
  for (i in 1:n_img) {
    path <- as.character(cluster_list_random_cl_1$image[i])
    img <- load.image(path)
    plot(img, axes = FALSE)
    title(main = paste("Cluster tSNE", cluster))
  }
}

# sapply(c(1), function(x) plot_random_images(cluster = 35))

plot_random_images_umap <- function(n_img = 9,
                                    cluster = 1,
                                    rows = 3,
                                    cols = 3) {
  cluster_list_random_cl_1 <- cluster_list_umap %>%
    #cluster_feature cluster_pca
    filter(cluster_feature == cluster) %>%
    sample_n(n_img,replace = TRUE)
  
  graphics::layout(matrix(c(1:n_img), rows, cols, byrow = TRUE))
  for (i in 1:n_img) {
    path <- as.character(cluster_list_random_cl_1$image[i])
    img <- load.image(path)
    plot(img, axes = FALSE)
    title(main = paste("Cluster umap", cluster))
  }
}

sapply(c(1), function(x) plot_random_images_umap(cluster = 4))

all_clusters<-c(1:30)

for(i in 1:length(all_clusters)){
 
  # par(mfrow=c(3,3))
  # plot_random_images_tsne(cluster = i)

  par(mfrow=c(3,3))
  plot_random_images_umap(cluster = i)

}
