# O QUE FAZER AINDA?
    # AUMENTAR BASE DE DADOS, GIRANDO IMAGENS OU MEXER ELAS 
    # AUMENTAR NUMERO DE CAMADAS
    # TECNICAS DE REGULARIZACAO PARA DEIXA A REDE MAIS INTELIGENTE
    
import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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

#Fazendo o "Dowload das imagens" para definir as que serão usadas para teste,validacao e treino
train_dataset = image_dataset_from_directory(train_folder, 
                                              image_size=(200,200),
                                              batch_size=32) #quanto em quanto o peso sera ajustado

validation_dataset = image_dataset_from_directory(val_folder, 
                                              image_size=(200,200),
                                              batch_size=32)

test_dataset = image_dataset_from_directory(test_folder, 
                                              image_size=(200,200),
                                              batch_size=32)

#Visualizar os batchs
for data_batch, labels_batch in train_dataset:
    print("data batch shape: ", data_batch.shape)
    print("labels batch shape: ", labels_batch.shape)
    print(data_batch[0].shape)
    break

#Criando e treinando modelo
model = keras.Sequential() #API Sequencial -> mais simples, tanto para apredizado quanto funcionamento
model.add (Rescaling(scale=1.0/255)) #Pegar cada cor da imagem e representar como 0 ou1 , para facilitar calculos etc.
model.add (Conv2D(32, kernel_size=(5,5), activation='relu')) #Permite extrair a geomatria que precisa do objeto para o reconhecimento
model.add (MaxPooling2D(pool_ize=(2,2))) #Reduzir a sobrecarga de informações, reduz a imagem, levando o maior valor de cor para frente a cada vez que ele "anda"
model.add (Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add (Flatten()) #Pegar todas as imagens e bota num vetor com todos os pixels
model.add (Dense(1, activation="sigmoid")) #Dizer se é ou não gado

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#Salva a melhor fase dos acertos da rede neural, sabe que é a melhor fase da rede por meio do numero de erros e acertos
callbacks = [  
    ModelCheckpoint(
        filepath="model1.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks
)

#Mostrar os resultados da rede em graficos para analisar estagnação ou overfiting
import matplotlib.pyplot as plt
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
plt.plot(epochs, loss, "b", label="Val loss")
plt.xlabel("Épocas")
plt.ylabel("%s")
plt.title("Loss de Treino e Validacao")
plt.legend()
plt.show()

model.summary()

#Resultado dos Conjuntos de Teste
from tenserflow import keras
model = keras.models.load_model("model1.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: `{test_acc:.3f}")

#Codigo para testar imagens baixadas por fora
#from matplotlib import pyplot as plt
