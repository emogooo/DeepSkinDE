from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from random import choice
from random import uniform
from numpy.random import randint
import matplotlib.pyplot as plt

train_datagen=ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('dataset/train', target_size=(224,224), batch_size=32, class_mode='categorical')
validation_set=test_datagen.flow_from_directory('dataset/validation', target_size=(224,224), batch_size=32, class_mode='categorical')
test_set=test_datagen.flow_from_directory('dataset/test', target_size=(224,224), batch_size=32, class_mode='categorical')


def initialization():  
  f1 = choice([16, 32, 64])
  f2 = choice([32, 64, 128])
  k = choice([3, 4, 5])
  d1 = choice([16, 32, 64, 128])
  d2 = choice([64, 128, 256])
  do1 = round(uniform(0.25, 0.8), 2)
  do2 = round(uniform(0.25, 0.8), 2)
  op = choice(["adamax", "adadelta", "adam", "adagrad"])
  ep = randint(7, 20)
  
  parameters = {}
  parameters["f1"] = f1
  parameters["f2"] = f2
  parameters["k"] = k
  parameters["d1"] = d1
  parameters["d2"] = d2
  parameters["do1"] = do1
  parameters["do2"] = do2
  parameters["op"] = op
  parameters["ep"] = ep
  return parameters

def generatePopulation(n):
  population = []
  for i in range(n):
    chromosome = initialization()
    population.append(chromosome)
  return population

def CNN_Model( f1, f2, k, d1, d2, do1, do2, op, ep):
  model = Sequential()
  
  model.add(Conv2D(filters = f1, kernel_size = (k, k), activation = "relu", input_shape = (224,224,3)))
  model.add(Conv2D(filters = f1, kernel_size = (k, k), activation = "relu"))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(2,2))
  
  model.add(Conv2D(filters = f2, kernel_size = (k, k), activation = "relu"))
  model.add(Conv2D(filters = f2, kernel_size = (k, k), activation = "relu"))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(2,2))
  
  model.add(Flatten())
  
  model.add(Dense(units = d1, activation = "relu"))
  model.add(Dropout(rate = do1))
  
  model.add(Dense(units = d2, activation = "relu"))
  model.add(Dropout(rate = do2))
  
  model.add(Dense(3, activation= "softmax"))

  model.compile(loss = "categorical_crossentropy", optimizer = op, metrics = ["accuracy"])
  model.fit(training_set, steps_per_epoch=training_set.n//training_set.batch_size, epochs=ep, validation_data=validation_set, validation_steps=validation_set.n//validation_set.batch_size)

  return model

def fitnessEvaluation(model):
  metrics = model.evaluate(test_set)
  return metrics[1]

def processFirstGeneration(populationSize):
    population = generatePopulation(populationSize)
    populationFitness = []
    for chromosome in population:
      f1 = chromosome["f1"]
      f2 = chromosome["f2"]
      k = chromosome["k"]
      d1 = chromosome["d1"]
      d2 = chromosome["d2"]
      do1 = chromosome["do1"]
      do2 = chromosome["do2"]
      op = chromosome["op"]
      ep = chromosome["ep"]
    
      acc=0
      try:
        model = CNN_Model(f1, f2, k, d1, d2, do1, do2, op, ep)
        acc = fitnessEvaluation(model)
        print("Parametreler: ", chromosome)
        print("Accuracy: ", round(acc,3), "\n")
      except:
        print("Parametreler: ", chromosome)
        print("Geçersiz parametere - Çalışma durduruldu\n")
      populationFitness.append(acc)
      
    print("İlk oluşumun değerleri: ", populationFitness)
    return population, populationFitness

generation = 100
populationSize = 6
accuracyHistory = list()
population, populationFitness = processFirstGeneration(populationSize)
firstBestAcc = max(populationFitness)

for i in range(generation):

    #Rastgele bir kromozom seçilmesi
    
    idxChr1 = randint(0, populationSize - 1)
    
    # İki farklı kromozomun seçilimi ve fark vektörünün oluşumu
      
    idxChr2 = randint(0, populationSize - 1)  
    
    while idxChr2 == idxChr1:
        idxChr2 = randint(0, populationSize - 1)  
    
    idxChr3 = randint(0, populationSize - 1)
    while idxChr3 == idxChr1 or idxChr3 == idxChr2:
        idxChr3 = randint(0, populationSize - 1) 
        
        
    differenceF1 = population[idxChr2]["f1"] - population[idxChr3]["f1"]
    differenceF2 = population[idxChr2]["f2"] - population[idxChr3]["f2"]
    differenceK = population[idxChr2]["k"] - population[idxChr3]["k"]
    differenceD1 = population[idxChr2]["d1"] - population[idxChr3]["d1"]
    differenceD2 = population[idxChr2]["d2"] - population[idxChr3]["d2"]
    differenceDO1 = population[idxChr2]["do1"] - population[idxChr3]["do1"]
    differenceDO2 = population[idxChr2]["do2"] - population[idxChr3]["do2"]
    differenceOp = choice([population[idxChr2]["op"], population[idxChr3]["op"]])
    differenceEp = population[idxChr2]["ep"] - population[idxChr3]["ep"]
    
    # Fark vektörünün optimize edilmesi
    
    differenceF1 = 1 if differenceF1 < 1 else 128 if differenceF1 > 128 else differenceF1
    differenceF2 = 1 if differenceF2 < 1 else 128 if differenceF2 > 128 else differenceF2
    differenceK = 2 if differenceK < 2 else 6 if differenceK > 6 else differenceK
    differenceD1 = 16 if differenceD1 < 16 else 256 if differenceD1 > 256 else differenceD1
    differenceD2 = 16 if differenceD2 < 16 else 256 if differenceD2 > 256 else differenceD2
    differenceDO1 = 0.1 if differenceDO1 < 0.1 else 0.75 if differenceDO1 > 0.75 else differenceDO1
    differenceDO2 = 0.1 if differenceDO2 < 0.1 else 0.75 if differenceDO2 > 0.75 else differenceDO2
    differenceEp = 6 if differenceEp < 6 else 20 if differenceEp > 20 else differenceEp
    
    
    # Fark vektörü ile toplanacak kromozomun seçilmesi (MUTASYON)
    
    idxChr4 = randint(0, populationSize - 1)
    while idxChr4 == idxChr1 or idxChr4 == idxChr2 or idxChr4 == idxChr3:
        idxChr4 = randint(0, populationSize - 1) 
    
    # Fark vektörü ile 4. kromozomun toplanması
    
    mutantF1 = population[idxChr4]["f1"] + differenceF1
    mutantF2 = population[idxChr4]["f2"] + differenceF2
    mutantK = population[idxChr4]["k"] + differenceK
    mutantD1 = population[idxChr4]["d1"] + differenceD1
    mutantD2 = population[idxChr4]["d2"] + differenceD2
    mutantDO1 = population[idxChr4]["do1"] + differenceDO1
    mutantDO2 = population[idxChr4]["do2"] + differenceDO2
    mutantOp = choice([population[idxChr4]["op"], differenceOp])
    mutantEp = population[idxChr4]["ep"] + differenceEp
    
    
    # Mutant kromozomun optimize edilmesi
    
    mutantF1 = 16 if mutantF1 < 16 else 128 if mutantF1 > 128 else mutantF1
    mutantF2 = 16 if mutantF2 < 16 else 128 if mutantF2 > 128 else mutantF2
    mutantK = 2 if mutantK < 2 else 6 if mutantK > 6 else mutantK
    mutantD1 = 16 if mutantD1 < 16 else 256 if mutantD1 > 256 else mutantD1
    mutantD2 = 16 if mutantD2 < 16 else 256 if mutantD2 > 256 else mutantD2
    mutantDO1 = 0.1 if mutantDO1 < 0.1 else 0.75 if mutantDO1 > 0.75 else mutantDO1
    mutantDO2 = 0.1 if mutantDO2 < 0.1 else 0.75 if mutantDO2 > 0.75 else mutantDO2
    mutantEp = 6 if mutantEp < 6 else 20 if mutantEp > 20 else mutantEp
    
    
    # Mutant kromozom ile ilk seçilen kromozomun eşlenmesi (ÇAPRAZLAMA)
    
    newF1 = choice([population[idxChr1]["f1"], mutantF1])
    newF2 = choice([population[idxChr1]["f2"], mutantF2])
    newK = choice([population[idxChr1]["k"], mutantK])
    newD1 = choice([population[idxChr1]["d1"], mutantD1])
    newD2 = choice([population[idxChr1]["d2"], mutantD2])
    newDO1 = choice([population[idxChr1]["do1"], mutantDO1])
    newDO2 = choice([population[idxChr1]["do2"], mutantDO2])
    newOp = choice([population[idxChr1]["op"], mutantOp])
    newEp = choice([population[idxChr1]["ep"], mutantEp])
    
    
    # Oluşturulan yeni kromozomun model olarak tanımlanması, eğitilmesi ve uygunluk değerinin hesaplanması
    
    newChromosome = {}
    newChromosome["f1"] = newF1
    newChromosome["f2"] = newF2
    newChromosome["k"] = newK
    newChromosome["d1"] = newD1
    newChromosome["d2"] = newD2
    newChromosome["do1"] = newDO1
    newChromosome["do2"] = newDO2
    newChromosome["op"] = newOp
    newChromosome["ep"] = newEp
    
    acc = 0
    try:
      model = CNN_Model(newF1, newF2, newK, newD1, newD2, newDO1, newDO2, newOp, newEp)
      acc = fitnessEvaluation(model)
      print("Parametreler: ", newChromosome)
      print("Accuracy: ", round(acc,3), "\n")
    except:
      print("Parametreler: ", newChromosome)
      print("Geçersiz parametere - Çalışma durduruldu\n")
    
    # Hesaplanan uygunluk değeri ilk seçilen kromozomdan daha iyiyse değişimin sağlanması
    
    if acc > populationFitness[idxChr1]:
        print("Daha iyi bir model üretildi.")
        print("Üretilen modelin accuracy değeri: ", acc)
        print("Önceki modelin accuracy değeri: ", populationFitness[idxChr1])
        print("Accuracy artış miktarı: ", round((acc - populationFitness[idxChr1]), 3))
        print("Accuracy artış oranı: %", round((acc - populationFitness[idxChr1]) / populationFitness[idxChr1], 3) * 100)
        populationFitness[idxChr1] = acc
        population[idxChr1] = newChromosome
        print("Yeni popülasyon değerleri: ", populationFitness)
    else:
        print("Gelişme sağlanamadı.")
        
    accuracyHistory.append(acc)
   
print("Üretilen en iyi modelin accuracy değeri: ", max(populationFitness))
print("İlk popülasyonun en iyi modelinin accuracy değeri: ", firstBestAcc)
print("Accuracy artış miktarı: ", round((max(populationFitness) - firstBestAcc), 3))
print("Accuracy artış oranı: %", round((max(populationFitness) - firstBestAcc) / firstBestAcc, 3) * 100)

maxFitness = max(populationFitness)
idx1 = populationFitness.index(populationFitness)
print("En iyi hiperparametreler: ", population[idx1])
   
plt.plot(accuracyHistory, color='Blue', marker='o',mfc='Red' )
plt.xticks(range(0,len(accuracyHistory)+1, 1))
plt.ylabel('Accuracy')
plt.xlabel('Generation')
plt.title("DGA ile Model Optimizasyon Grafiği")
plt.savefig("C:\Projects\DK\DGA_Acc_History")
    