# O QUE FAZER AINDA?
    # AUMENTAR BASE DE DADOS, GIRANDO IMAGENS OU MEXER ELAS 
    # AUMENTAR NUMERO DE CAMADAS
    # TECNICAS DE REGULARIZACAO PARA DEIXA A REDE MAIS INTELIGENTE
    
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import ModelCheckpoint



#Verificando versão do tensorflow-GPU
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

#Criando pastas para receber imagens
current_dir = "E:\IMAGENS_DATASETS"

train_folder = current_dir + "\\Treino"
val_folder = current_dir + "\\Validacao" 
test_folder = current_dir + "\\Test"

tamanho_train_folder_gados = len(os.listdir(os.path.join(train_folder,'gado')))
tamanho_train_folder_semgados = len(os.listdir(os.path.join(train_folder,'semgado')))

tamanho_val_folder_gados = len(os.listdir(os.path.join(val_folder,'gado')))
tamanho_val_folder_semgados = len(os.listdir(os.path.join(val_folder,'semgado')))

tamanho_test_folder_gados = len(os.listdir(os.path.join(test_folder,'gado2')))
tamanho_test_folder_semgados = len(os.listdir(os.path.join(test_folder,'semgado2')))

print('Treino Gado: %s' % tamanho_train_folder_gados)
print('Treino Sem Gado: %s' % tamanho_train_folder_semgados)

print('Validacao Gado: %s' % tamanho_val_folder_gados)
print('Validacao Sem Gado: %s' % tamanho_val_folder_semgados)

print('Teste Gado: %s' % tamanho_test_folder_gados)
print('Teste Sem Gado: %s' % tamanho_test_folder_semgados)

image_width = 160
image_height = 160
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shpae = image_size + (image_color_channel,)

batch_size = 32
epochs = 50
learning_rate = 0.0001

class_names = ['SEMGADO', 'COMGADO']

#Fazendo o "Dowload das imagens" para definir as que serão usadas para teste,validacao e treino
train_dataset = image_dataset_from_directory(train_folder, 
                                            image_size = image_size,
                                            batch_size = batch_size,) #quanto em quanto o peso sera ajustado

validation_dataset = image_dataset_from_directory(val_folder, 
                                            image_size = image_size,
                                            batch_size = batch_size)

test_dataset = image_dataset_from_directory(test_folder, 
                                            image_size = image_size,
                                            batch_size = batch_size)

#Visualizar os batchs
for data_batch, labels_batch in train_dataset:
    print("data batch shape: ", data_batch.shape)
    print("labels batch shape: ", labels_batch.shape)
    print(data_batch[0].shape)
    break

#Modelo ALEXNET
# model = keras.Sequential() 

# model.add (Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')) 
# model.add (MaxPooling2D(pool_size=3,strides=2)) 

# model.add (Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')) 
# model.add (MaxPooling2D(pool_size=3,strides=2)) 

# model.add (Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')) 
# model.add (Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')) 
# model.add (Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')) 
# model.add (MaxPooling2D(pool_size=3,strides=2)) 

# model.add (Flatten()) 
# model.add (Dense(4096, activation="relu"))
# model.add (Dropout(0.5))
# model.add (Dense(4096, activation="relu"))
# model.add (Dropout(0.5))

# model.add (Dense(10))

#Criando meu modelo e treinando ele
model = keras.Sequential() #API Sequencial -> mais simples, tanto para apredizado quanto funcionamento

model.add (Rescaling(scale=1.0/255)) #Pegar cada cor da imagem e representar como 0 ou1 , para facilitar calculos etc.

model.add (Conv2D(32, kernel_size=(3,3), activation='relu')) #Permite extrair a geomatria que precisa do objeto para o reconhecimento
model.add (MaxPooling2D(pool_size=(2,2))) #Reduzir a sobrecarga de informações, reduz a imagem, levando o maior valor de cor para frente a cada vez que ele "anda"

model.add (Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Dropout(0.5))#Evitar overfitting

model.add (Flatten()) #Pegar todas as imagens e bota num vetor com todos os pixels
model.add (Dense(256, activation="relu"))
model.add (Dropout(0.5)) 
model.add (Dense(1, activation="sigmoid")) #Dizer se é ou não gado

#model_save = pickle.dumps(model)

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
              metrics=["accuracy"])

history = model.fit(
     train_dataset,
     epochs=epochs,
     validation_data=validation_dataset,
 )

# #Mostrar os resultados da rede em graficos para analisar estagnação ou overfiting

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "r", label="Treino acc")
plt.plot(epochs, val_accuracy, "b", label="Val acc")
plt.xlabel("Épocas")
plt.ylabel("%s")
plt.title("Acurácia de Treino e Validacao")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r", label="Treino loss")
plt.plot(epochs, val_loss, "b", label="Val loss")
plt.xlabel("Épocas")
plt.ylabel("%s")
plt.title("Loss de Treino e Validacao")
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(test_dataset)
print(f"Acuracia Teste: {test_acc:.3f}")

#model.summary()

def plot_dataset_predictions(dataset) :
    
    features, labels = dataset.as_numpy_iterator().next()
    
    predictions = model.predict_on_batch(features).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)
    
    print('Labels-> %s' % labels)
    print('Predictions-> %s' % predictions.numpy())
    
plot_dataset_predictions(test_dataset)


# Exemplo de rótulos verdadeiros e preditos
y_true = [1, 0, 0, 1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 1, 0, 0, 1, 0]

# Criar a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Visualizar a matriz de confusão
labels = np.unique(y_true)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Rótulo Previsto')
plt.ylabel('Rótulo Verdadeiro')

# Adicionar os valores numericos nas células
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.show()


# model.save('path/to/model')
# model = tf.keras.models.load_model('path/to/model')
