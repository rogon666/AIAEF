#------------------------------------------------------------------------------
# Modelo de deep learning para vision por computadora, basado en Keras
# Rolando Gonzales Martinez
# Agosto 2023
#------------------------------------------------------------------------------

library(keras)
mnist <- dataset_mnist()
x_entrenamiento <- mnist$train$x
y_entrenamiento <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# preparando los datos
x_entrenamiento <- array_reshape(x_entrenamiento, c(nrow(x_entrenamiento), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_entrenamiento <- x_entrenamiento/255
x_test <- x_test/255
y_entrenamiento <- to_categorical(y_entrenamiento, 10)
y_test <- to_categorical(y_test, 10)

# Arquitectura del modelo_DL:
modelo_DL <- keras_model_sequential() 
modelo_DL %>% 
  layer_dense(units = 256, activation = 'softmax', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'softmax') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(modelo_DL)

# Definiendo funciones objetivo:
modelo_DL %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Estimando (entrenando) el modelo:
history <- modelo_DL %>% fit(
  x_entrenamiento, y_entrenamiento, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
plot(history)

# Evaluando el modelo DL:
modelo_DL %>% evaluate(x_test, y_test)

# Realizando predicciones:
modelo_DL %>% predict(x_test) %>% k_argmax()

# -----------------------------------------------------------------------------