data <- read.csv("train.csv")
data <- textshape::column_to_rownames(data, loc=1) # Paso el ID de la vivienda como índice de fila

set.seed(2) 

library(ggplot2) # Para gráficas personalizadas
library(dplyr)
library(caret)    # Para crear particiones estratificadas
library(stats)    # Para PCA y regresión lineal
library(glmnet)   # Necesitamos esta librería para Ridge y Lasso

#-------------------------------------- ANÁLISIS EXPLORATORIO DE DATOS --------------------------------------

# Divido las variables en categóricas y numéricas
var_num <- names(select(.data = data, where(is.numeric)))
var_cat <- names(select(.data = data, where(is.character)))

# Vamos a realizar un análisis de las variables para entender los datos que tenemos

# Variables numéricas
par(mfrow = c(1,2))
for (i in var_cat) {
  boxplot(data$SalePrice*1e-5 ~ data[[i]],
          main = paste("SalePrice vs", i),
          xlab = i,
          ylab = "SalePrice (cien miles USD)",
          col = "lightblue")
  
  barplot(table(data[[i]]),
          main = paste("Barplot de", i),
          xlab = "Categoría",
          ylab = "Frecuencia",
          col = "darkred")
}

par(mfrow = c(1,2))
# Análisis de las variables numéricas
for(i in var_num){
  plot(data$SalePrice*1e-5 ~ data[[i]],
       main = paste("SalePrice vs", i),
       xlab = i,
       ylab = "SalePrice (cien miles USD)",
       col = "blue")
  
  hist(data[[i]],
       main = paste("Histograma de", i),
       xlab = i,
       ylab = "Frecuencia",
       col = "red")
}

# Veamos qué variables tienen datos faltantes y cuántos son
NAs <- colSums(is.na(data))
NAs <- NAs[NAs != 0]
sort(NAs/nrow(data), decreasing = TRUE)

# Voy a eliminar directamente aquellas variables que tienen una proporcion de valores faltantes superior al 90%
# Esto porque son tantas que en sí misma ya forman una propia categoría y no añaden información extra

data$Alley <- NULL
data$PoolQC <- NULL
data$MiscFeature <- NULL

# Los NAs de las columnas Garage se deben a viviendas que no tienen garaje. Lo que voy a hacer es crear
# Una nueva categoría "No" para aquellos casos en los que esto suceda.
data$GarageType[is.na(data$GarageType)] <- "No"
data$GarageQual[is.na(data$GarageQual)] <- "No"
data$GarageCond[is.na(data$GarageCond)] <- "No"
data$GarageYrBlt[is.na(data$GarageYrBlt)] <-  0 # Ojo que esto es una variable numérica. Voy a imputarlo con
                                                # un 0 para identificar que no tiene garaje
data$GarageFinish[is.na(data$GarageFinish)] <- "No"
data$GarageQual[is.na(data$GarageQual)] <-  "No"

# Lo mismo para Fence
data$Fence[is.na(data$Fence)] <- "No"

# Lo mismo ocurre con las columnas Bsmt
data$BsmtQual[is.na(data$BsmtQual)] <- "No"
data$BsmtCond[is.na(data$BsmtCond)] <- "No"
data$BsmtExposure[is.na(data$BsmtExposure)] <- "NoBase"
data$BsmtFinType1[is.na(data$BsmtFinType1)] <- "No"
data$BsmtFinType2[is.na(data$BsmtFinType2)] <- "No"

# Y lo mismo ocurre con FirePlaceQu
data$FireplaceQu[is.na(data$FireplaceQu)] <- "No"

# A partir de aquí, los datos faltantes ya no tienen un sentido lógico, sino que se deben a errores

#---------------CREACIÓN DE LOS CONJUNTOS DE ENTRENAMIENTO, VALIDACIÓN Y TEST-------------------------------

# Voy a diferenciar las variables predictoras de la variable objetivo
X <- data[, 1:(ncol(data)-1)]
y <- log(data[, ncol(data)])

# Voy a dividir mi conjunto de datos en entrenamiento, validación y prueba
# Paso 1: Separar 60% para entrenamiento
# createDataPartition hace división estratificada respetando distribución de y
trainIndex <- createDataPartition(y, p = 0.6, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]

# Paso 2: Del 40% restante, dividir en 50%-50% (validación y reserva/test)
remainingIndex <- setdiff(seq_len(nrow(data)), trainIndex)
valIndex <- createDataPartition(y[remainingIndex], p = 0.5, list = FALSE)

X_val <- X[remainingIndex[valIndex], ]
y_val <- y[remainingIndex[valIndex]]

X_test <- X[remainingIndex[-valIndex], ]  # Renombrado a "test" 
y_test <- y[remainingIndex[-valIndex]]

# Vamos a continuar con el tratamiento de datos faltantes 

# Para Lot_Frontage, imputaré con la media
mean_Lot_frontage <- mean(X_train$LotFrontage, na.rm = TRUE)
X_train$LotFrontage[is.na(X_train$LotFrontage)] <- mean_Lot_frontage
X_test$LotFrontage[is.na(X_test$LotFrontage)] <- mean_Lot_frontage
X_val$LotFrontage[is.na(X_val$LotFrontage)] <- mean_Lot_frontage

moda_MasVnrType <- names(sort(table(X_train$MasVnrType), decreasing = TRUE))[1]
# La moda de MasVnrType es "None", que se corresponde con un 0 (no revestimiento) en MasVnrArea
X_train$MasVnrType[is.na(X_train$MasVnrType)] <- moda_MasVnrType
X_test$MasVnrType[is.na(X_test$MasVnrType)] <- moda_MasVnrType
X_val$MasVnrType[is.na(X_val$MasVnrType)] <- moda_MasVnrType

X_train$MasVnrArea[is.na(X_train$MasVnrArea)] <- 0
X_test$MasVnrArea[is.na(X_test$MasVnrArea)] <- 0
X_val$MasVnrArea[is.na(X_val$MasVnrArea)] <- 0


NAs <- colSums(is.na(X_train))
NAs <- NAs[NAs != 0]
NAs

# Ahora ya solo tenemos un valor faltante en el conjunto de entrenamiento. Lo imputaremos por su moda
X_train$Electrical[is.na(X_train$Electrical)] <- names(sort(table(X_train$Electrical), decreasing = TRUE))[1]

anyNA(rbind(X_train, X_test, X_val)) # Ya no tenemos ningún dato faltante

#------------------------------------ENCODING DE VARIABLES CATEGÓRICAS-------------------------------------------

# Voy a identificar cuales de mis variables categóricas siguen un orden (básicamente todas las que se refieren a la calidad o estado de algo).
for (i in var_cat) {
  cat(i, "-------->", unique(X_train[[i]]), "\n")
}

var_cat_si_orden <- c("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond")

# Ahora las convierto a nominal

for (i in var_cat_si_orden) {
  X_train[[i]] <- as.numeric(factor(X_train[[i]], levels = c("No", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)) - 1
  X_test[[i]] <- as.numeric(factor(X_test[[i]], levels = c("No", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)) - 1
  X_val[[i]] <- as.numeric(factor(X_val[[i]], levels = c("No", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)) - 1
}

# A BsmtExposure le hacemos un tratamiento a parte porque los niveles no coinciden con el resto

X_train$BsmtExposure <- as.numeric(factor(X_train$BsmtExposure, levels = c("NoBase", "No", "Mn", "Av", "Gd"), ordered = TRUE)) - 1
X_test$BsmtExposure <- as.numeric(factor(X_test$BsmtExposure, levels = c("NoBase", "No", "Mn", "Av", "Gd"), ordered = TRUE)) - 1
X_val$BsmtExposure <- as.numeric(factor(X_val$BsmtExposure, levels = c("NoBase", "No", "Mn", "Av", "Gd"), ordered = TRUE)) - 1

# El resto de variables categóricas ya no siguen un orden. Las codificaré mediante la técnica mean encoding y frequency encoding
# El problema de estas técnicas es que, puede haber categorías que solo existan en validación o test. La estrategia que emplearé si 
# esto ocurre es que las imputaré con la media global o con 0 en el caso de las frecuencias


# Voy a codificar los datos de Fence mediante sus frecuencias ya que hay en torno al 80% de viviendas sin Fence e imputar con la media podría no ser la mejor opción
freq_Fence <- sort(table(X_train$Fence), decreasing = TRUE) #Creo la tabla con las frecuencias con las que aparece cada categoría
#Mapeamos estas frecuencias a cada conjunto de datos
X_train$Fence <- freq_Fence[X_train$Fence] 
X_test$Fence <- freq_Fence[X_test$Fence]
X_val$Fence <- freq_Fence[X_val$Fence]

anyNA(c(X_train$Fence, X_test$Fence, X_val$Fence)) # Con esto me aseguro que no hay NAs en ningún conjunto


#El resto de datos los cofico mediante la media

var_cat <- names(select(.data = X_train, where(is.character)))

medias_list <- list() # Creo la lista donde iré guardando las medias asociadas a cada categoría de cada variable categórica restante
media_global <- mean(y_train) # Calculo la media global 

for (i in var_cat) {
  medias <- tapply(y_train, X_train[[i]], mean) # Creo el vector de medias para cada categoría de cada variable categórica
  medias_list[[i]] <- medias # Lo guardo en la lista
  
  X_train[[i]] <- medias[X_train[[i]]] # Mapeo la media de cada categoria
  
  # Validación. Ojo, hay que hacerlo con las medias calculadas en el set de entrenamiento
  X_val[[i]] <- medias[X_val[[i]]]
  X_val[[i]][is.na(X_val[[i]])] <- media_global
  
  # Repito lo mismo para el set de prueba
  X_test[[i]] <- medias[X_test[[i]]]
  X_test[[i]][is.na(X_test[[i]])] <- media_global
}

anyNA(rbind(X_train, X_test, X_val)) # Ahora ya no tenemos ningún valor faltante

# Aplico el logaritmo a las matrices debido a la enorme cantidad de outliers que hay en el data set; de esta forma intento reducir el efecto de los outliers.
# Como hay algunos valores que pueden ser 0, voy a aplicar el log(1 + x)
X_train <- as.matrix(log1p(X_train)) #Los transformo a matriz porque algunas funciones funcionan con matrices
X_test <- as.matrix(log1p(X_test))
X_val <- as.matrix(log1p(X_val))
#-----------------------------------------------------ESCALADO DE DATOS---------------------------------------------------------

# Una vez limpiado los datos, es hora de hacer el escalado de datos para un mejor desempeño del modelo
X_train_scaled <- scale(x = X_train, center = TRUE, scale = TRUE) # Me lleva los datos a una distribución de media 0 y std de 1
medias <- attr(X_train_scaled, "scaled:center")
standard_dev <- attr(X_train_scaled, "scaled:scale")

# Escalo también los conjuntos de validacion y prueba usando los parámetros calculados del conjunto de entrenamiento

X_test_scaled <- scale(X_test, center = medias, scale = standard_dev)
X_val_scaled <- scale(X_val, center = medias, scale = standard_dev)

#--------------------------------------------------------ANÁLISIS PCA-------------------------------------------------------------------

pca <- prcomp(X_train_scaled, scale. = FALSE, center = FALSE) # Hacemos el análisis PCA sobre el conjunto de entrenamiento escalado
var_acum <- summary(pca)$importance[3, ]

# Vamos a visualizar los datos obtenidos para tener una idea de lo que tenemos

pp <- double() # Voy a crear un vector para ver cómo va disminuyendo el aumento proporcionado por cada componente
for (i in 2:length(var_acum)) {
  pp[i-1] <- (var_acum[i] - var_acum[i-1]) 
  cat(i, var_acum[i-1], "------->", var_acum[i] - var_acum[i-1], "\n")
}
var_acum
par(mfrow = c(1, 3))
barplot(summary(pca)$importance[2, ], 
        main = "Varianza por Componente",
        xlab = "Componente Principal", 
        ylab = "Proporción de Varianza",
        col = "steelblue")

plot(var_acum, type = "b", 
     main = "Varianza Acumulada",
     xlab = "Número de Componentes", 
     ylab = "Proporción Acumulada",
     col = "darkred", lwd = 2)
#abline(h = 0.95, col = "blue", lty = 2, lwd = 2)
#legend("bottomright", legend = "Umbral 95%", 
#col = "blue", lty = 2, lwd = 2)

plot(pp, type = "b",
     main = "Disminución de la varianza acumulada por componente añadido",
     xlab = "Número de componentes",
     ylab = "Dismunción",
     col = "green")

# Seleccionar componentes que expliquen al menos 95% de la varianza
n_comp <- min(which(var_acum >= 0.95))

X_train_pca <- pca$x[,1:n_comp, drop = FALSE] # Me quedo con las 50 primeras componentes del PCA

# Ahora voy a hacer el cambio de base de los conjuntos de validación y prueba

X_test_pca <- as.matrix(X_test_scaled) %*% pca$rotation[, 1:n_comp, drop = FALSE]
X_val_pca <- as.matrix(X_val_scaled) %*% pca$rotation[, 1:n_comp, drop = FALSE]

# Ahora vamos a entrenar al modelo

modelo_pca <- lm(y_train ~ ., data = as.data.frame(X_train_pca))
summary(modelo_pca) # Con esto visualizamos algunos parámetros del entrenamiento

# Una vez entrenado el modelo, evaluamos en el conjunto de prueba y validación

pred_train_pca <- predict(modelo_pca, newdata = as.data.frame(X_train_pca))
pred_val_pca <- predict(modelo_pca, newdata = as.data.frame(X_val_pca))
pred_test_pca <- predict(modelo_pca, newdata = as.data.frame(X_test_pca))

# Ahora vamos a medir qué tan bueno es nuestro modelo
# Primero vamos a definir las métricas de rendimiento

rmse <- function(y_true, y_pred) sqrt(mean((y_true - y_pred)^2))
mse <- function(y_true, y_pred) mean((y_true - y_pred)**2)
mae <- function(y_true, y_pred) mean(abs(y_true - y_pred))
r2 <- function(y_true, y_pred) cor(y_true, y_pred)^2
r2_adj <- function(y_true, y_pred, k) {
  n <- length(y_true)
  r2 <- cor(y_true, y_pred)^2
  1 - ((1 - r2) * (n - 1)) / (n - k - 1)
}

# Vamos a visualizar ahora qué tan buenas han sido las predicciones de nuestro modelo
# # Predicciones vs Valores Reales
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))
plot(y_train, pred_train_pca, main = "Entrenamiento", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

plot(y_val, pred_val_pca, main = "Validación", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "green")
abline(0, 1, col = "red", lwd = 2)

plot(y_test, pred_test_pca, main = "Test", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
mtext("Modelo lineal con PCA", outer = TRUE, cex = 1.5)

# Vamos a probar ahora los modelos de regularizaión Lasso y Ridge

#----------------------------RIDGE-------------------

# Para obtener el mejor parámetro lambda haremos una cross validation

lambda_grid <- 10^seq(2, -2, length = 1000)

ridge_cv_pca <- cv.glmnet(
  x = X_train_pca,           # Datos de entrenamiento (componentes PCA)
  y = y_train,               # Variable objetivo
  alpha = 0,             # Ridge (L2)
  lambda = lambda_grid,  # Grilla de lambdas a probar
  nfolds = 10,           # 10-fold cross-validation
  standardize = FALSE    # Ya escalamos con PCA
)
lambda_ridge_opt_pca <- ridge_cv_pca$lambda.min # Guardamos el mejor valor que nos ofrece validación cruzada

# Una vez encontrado el lambda óptimo, entrenamos el modelo Ridge 

modelo_ridge_pca <- glmnet(
  x = X_train_pca,
  y = y_train,
  alpha = 0,
  lambda = lambda_ridge_opt_pca,
  standardize = FALSE
)


# Ahora hacemos el modelo Ridge sin aplicar PCA sobre los datos

ridge_cv <- cv.glmnet(
  x = X_train_scaled,           # Datos de entrenamiento (componentes PCA)
  y = y_train,               # Variable objetivo
  alpha = 0,             # Ridge (L2)
  lambda = lambda_grid,  # Grilla de lambdas a probar
  nfolds = 10,           # 10-fold cross-validation
  standardize = FALSE    # Ya escalamos con PCA
)
lambda_ridge_opt <- ridge_cv$lambda.min # Guardamos el mejor valor que nos ofrece validación cruzada

# Una vez encontrado el lambda óptimo, entrenamos el modelo Ridge 

modelo_ridge <- glmnet(
  x = X_train_scaled,
  y = y_train,
  alpha = 0,
  lambda = lambda_ridge_opt,
  standardize = FALSE
)

# Hacemos exactamente lo mismo para Lasso

lambda_grid <- 10^seq(2, -2, length = 1000)

lasso_cv_pca <- cv.glmnet(
  x = X_train_pca,           # Datos de entrenamiento (componentes PCA)
  y = y_train,               # Variable objetivo
  alpha = 1,             # Lasso (L1)
  lambda = lambda_grid,  # Grilla de lambdas a probar
  nfolds = 10,           # 10-fold cross-validation
  standardize = FALSE    # Ya escalamos con PCA
)
lambda_lasso_opt_pca <- lasso_cv_pca$lambda.min # Guardamos el mejor valor que nos ofrece validación cruzada

# Una vez encontrado el lambda óptimo, entrenamos el modelo Lasso 

modelo_lasso_pca <- glmnet(
  x = X_train_pca,
  y = y_train,
  alpha = 1,
  lambda = lambda_lasso_opt_pca,
  standardize = FALSE
)

# Ahora hacemos Lasso pero sin aplicar PCA sobre el conjunto de entrenamiento
lambda_grid <- 10^seq(2, -2, length = 1000)

lasso_cv <- cv.glmnet(
  x = X_train_scaled,           # Datos de entrenamiento (componentes PCA)
  y = y_train,               # Variable objetivo
  alpha = 1,             # Lasso (L1)
  lambda = lambda_grid,  # Grilla de lambdas a probar
  nfolds = 10,           # 10-fold cross-validation
  standardize = FALSE    # Ya escalamos con PCA
)
lambda_lasso_opt <- lasso_cv$lambda.min # Guardamos el mejor valor que nos ofrece validación cruzada

# Una vez encontrado el lambda óptimo, entrenamos el modelo Lasso 

modelo_lasso <- glmnet(
  x = X_train_scaled,
  y = y_train,
  alpha = 1,
  lambda = lambda_lasso_opt,
  standardize = FALSE
)

#--------------------------------------------PREDICCIONES RIDGE----------------------------------------------

# Modelo Ridge con PCA
pred_train_ridge_pca <- as.vector(predict(modelo_ridge_pca, newx = X_train_pca, s = lambda_ridge_opt_pca))
pred_val_ridge_pca <- as.vector(predict(modelo_ridge_pca, newx = X_val_pca, s = lambda_ridge_opt_pca))
pred_test_ridge_pca <- as.vector(predict(modelo_ridge_pca, newx = X_test_pca, s = lambda_ridge_opt_pca))

# Vamos a visualizar ahora qué tan buenas han sido las predicciones de nuestro modelo
# # Predicciones vs Valores Reales
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))
plot(y_train, pred_train_ridge_pca, main = "Entrenamiento", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

plot(y_val, pred_val_ridge_pca, main = "Validación", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "green")
abline(0, 1, col = "red", lwd = 2)

plot(y_test, pred_test_ridge_pca, main = "Test", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
mtext("Modelo Ridge con PCA", outer = TRUE, cex = 1.5)

# Modelo Ridge sin PCA
pred_train_ridge <- as.vector(predict(modelo_ridge, newx = X_train_scaled, s = lambda_ridge_opt))
pred_val_ridge <- as.vector(predict(modelo_ridge, newx = X_val_scaled, s = lambda_ridge_opt))
pred_test_ridge <- as.vector(predict(modelo_ridge, newx = X_test_scaled, s = lambda_ridge_opt))

# Vamos a visualizar ahora qué tan buenas han sido las predicciones de nuestro modelo
# # Predicciones vs Valores Reales
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))
plot(y_train, pred_train_ridge, main = "Entrenamiento", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

plot(y_val, pred_val_ridge, main = "Validación", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "green")
abline(0, 1, col = "red", lwd = 2)

plot(y_test, pred_test_ridge, main = "Test", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
mtext("Modelo Ridge sin PCA", outer = TRUE, cex = 1.5)

#--------------------------------------------PREDICCIONES LASSO----------------------------------------------

# Modelo Lasso con PCA
pred_train_lasso_pca <- as.vector(predict(modelo_lasso_pca, newx = X_train_pca, s = lambda_lasso_opt_pca))
pred_val_lasso_pca <- as.vector(predict(modelo_lasso_pca, newx = X_val_pca, s = lambda_lasso_opt_pca))
pred_test_lasso_pca <- as.vector(predict(modelo_lasso_pca, newx = X_test_pca, s = lambda_lasso_opt_pca))

# Vamos a visualizar ahora qué tan buenas han sido las predicciones de nuestro modelo
# # Predicciones vs Valores Reales
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))
plot(y_train, pred_train_lasso_pca, main = "Entrenamiento", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

plot(y_val, pred_val_lasso_pca, main = "Validación", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "green")
abline(0, 1, col = "red", lwd = 2)

plot(y_test, pred_test_lasso_pca, main = "Test", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
mtext("Modelo Lasso con PCA", outer = TRUE, cex = 1.5)

# Modelo Lasso sin PCA
pred_train_lasso <- as.vector(predict(modelo_lasso, newx = X_train_scaled, s = lambda_lasso_opt))
pred_val_lasso <- as.vector(predict(modelo_lasso, newx = X_val_scaled, s = lambda_lasso_opt))
pred_test_lasso <- as.vector(predict(modelo_lasso, newx = X_test_scaled, s = lambda_lasso_opt))

# Vamos a visualizar ahora qué tan buenas han sido las predicciones de nuestro modelo
# # Predicciones vs Valores Reales
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))
plot(y_train, pred_train_lasso, main = "Entrenamiento", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

plot(y_val, pred_val_lasso, main = "Validación", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "green")
abline(0, 1, col = "red", lwd = 2)

plot(y_test, pred_test_lasso, main = "Test", 
     xlab = "ln(Real)", ylab = "ln(Predicho)", pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
mtext("Modelo Lasso sin PCA", outer = TRUE, cex = 1.5)

# Creamos una tabla con todas las métricas
resultados <- data.frame(
  Modelo = rep(c("Lineal", "Ridge_PCA", "Lasso_PCA", "Ridge", "Lasso"), each = 3),
  Conjunto = rep(c("Train", "Val", "Test"), 5),
  
  MSE = round(c(
    mse(y_train, pred_train_pca), mse(y_val, pred_val_pca), mse(y_test, pred_test_pca),
    mse(y_train, pred_train_ridge_pca), mse(y_val, pred_val_ridge_pca), mse(y_test, pred_test_ridge_pca),
    mse(y_train, pred_train_lasso_pca), mse(y_val, pred_val_lasso_pca), mse(y_test, pred_test_lasso_pca),
    mse(y_train, pred_train_ridge), mse(y_val, pred_val_ridge), mse(y_test, pred_test_ridge),
    mse(y_train, pred_train_lasso), mse(y_val, pred_val_lasso), mse(y_test, pred_test_lasso)
  ), 2),
  
  MAE = round(c(
    mae(y_train, pred_train_pca), mae(y_val, pred_val_pca), mae(y_test, pred_test_pca),
    mae(y_train, pred_train_ridge_pca), mae(y_val, pred_val_ridge_pca), mae(y_test, pred_test_ridge_pca),
    mae(y_train, pred_train_lasso_pca), mae(y_val, pred_val_lasso_pca), mae(y_test, pred_test_lasso_pca),
    mae(y_train, pred_train_ridge), mae(y_val, pred_val_ridge), mae(y_test, pred_test_ridge),
    mae(y_train, pred_train_lasso), mae(y_val, pred_val_lasso), mae(y_test, pred_test_lasso)
  ), 2),
  
  RMSE = round(c(
    rmse(y_train, pred_train_pca), rmse(y_val, pred_val_pca), rmse(y_test, pred_test_pca),
    rmse(y_train, pred_train_ridge_pca), rmse(y_val, pred_val_ridge_pca), rmse(y_test, pred_test_ridge_pca),
    rmse(y_train, pred_train_lasso_pca), rmse(y_val, pred_val_lasso_pca), rmse(y_test, pred_test_lasso_pca),
    rmse(y_train, pred_train_ridge), rmse(y_val, pred_val_ridge), rmse(y_test, pred_test_ridge),
    rmse(y_train, pred_train_lasso), rmse(y_val, pred_val_lasso), rmse(y_test, pred_test_lasso)
  ), 2),
  
  R_2 = round(c(
    r2(y_train, pred_train_pca), r2(y_val, pred_val_pca), r2(y_test, pred_test_pca),
    r2(y_train, pred_train_ridge_pca), r2(y_val, pred_val_ridge_pca), r2(y_test, pred_test_ridge_pca),
    r2(y_train, pred_train_lasso_pca), r2(y_val, pred_val_lasso_pca), r2(y_test, pred_test_lasso_pca),
    r2(y_train, pred_train_ridge), r2(y_val, pred_val_ridge), r2(y_test, pred_test_ridge),
    r2(y_train, pred_train_lasso), r2(y_val, pred_val_lasso), r2(y_test, pred_test_lasso)
  ), 2),
  
  R2_ajustado = round(c(
    r2_adj(y_train, pred_train_pca, ncol(X_train_pca)), 
    r2_adj(y_val, pred_val_pca, ncol(X_train_pca)), 
    r2_adj(y_test, pred_test_pca, ncol(X_train_pca)),
    
    r2_adj(y_train, pred_train_ridge_pca, ncol(X_train_pca)), 
    r2_adj(y_val, pred_val_ridge_pca, ncol(X_train_pca)), 
    r2_adj(y_test, pred_test_ridge_pca, ncol(X_train_pca)),
    
    r2_adj(y_train, pred_train_lasso_pca, ncol(X_train_pca)), 
    r2_adj(y_val, pred_val_lasso_pca, ncol(X_train_pca)), 
    r2_adj(y_test, pred_test_lasso_pca, ncol(X_train_pca)),
    
    r2_adj(y_train, pred_train_ridge, ncol(X_train_scaled)), 
    r2_adj(y_val, pred_val_ridge, ncol(X_train_scaled)), 
    r2_adj(y_test, pred_test_ridge, ncol(X_train_scaled)),
    
    r2_adj(y_train, pred_train_lasso, ncol(X_train_scaled)), 
    r2_adj(y_val, pred_val_lasso, ncol(X_train_scaled)), 
    r2_adj(y_test, pred_test_lasso, ncol(X_train_scaled))
  ), 2)
  
)
