from datetime import datetime

import cv2 as cv
import functions
import os

cam = cv.VideoCapture(0) #Iniciando a WebCam
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv.CascadeClassifier(f"{cv.haarcascades}{os.sep}{file_name}") #Modelo para reconhecer faces

dataframe = functions.load_dataframe() #Carregando dataframe com as imagens para treinamento

X_train, y_train = functions.train_test(dataframe) #Dividindo conjuntos de treino e teste
pca = functions.pca_model(X_train) #Modelo PCA para extração de features da imagem

X_train = pca.transform(X_train) #Conjunto de treino com features extraídas
#X_test = pca.transform(X_test) #Conjunto de teste com features extraídas

knn = functions.knn(X_train, y_train) #Treinando modelo classificatório KNN

#Rótulo das classificações
label = {
    0: "Sem mascara",
    1: "Com mascara"
}

#Abrindo a webcam...
while True:
    status, frame = cam.read() #Lendo a imagem e extraindo frame

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    #Transformando a imagem em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detectando faces na imagem
    faces = classifier.detectMultiScale(gray)

    #Iterando nas faces encontradas
    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w] #Recortando região da face

        # if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
        gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
        vector = pca.transform([gray_face.flatten()]) #Extraindo features da imagem

        pred = knn.predict(vector)[0] #Classificando a imagem
        classification = label[pred]

        #Desenhando retangulos em torno da face
        if pred == 0:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
            rosto = gray_face

            # cv.imwrite("fotos/fotos." + f'{datetime.today().strftime("%d-%m-%Y %H:%M:%S")}' + ".jpg", rosto)
            print("Mascara não identificada")
        elif pred == 1:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

            numero = int(functions.count()[-1])
            while numero in functions.count():
                numero += 1
            cv.imwrite(f"imagens/maskon/{numero}.png", gray_face)
            # with open('logs/fots','a') as f:
            #     f.write()
            print("Mascara identificada")
            cv.waitKey(1)
            cam.release()
            cv.destroyAllWindows()
            
            #Escrevendo classificação e quantidade de faces vistas
            cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(frame, f"{len(faces)} rostos identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    #Mostrando o frame
    cv.imshow("Cam", frame)
