import cv2
import os
import cv2.ml
import numpy as np
from imutils.object_detection import non_max_suppression

detectarFace=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50,threshold = 0)
#fisherface = cv2.face.FisherFaceRecognizer_create()
#lbphface = cv2.face.LBPHFaceRecognizer_create()
largura,altura = 220,220
fonte = cv2.FONT_HERSHEY_COMPLEX
camera = cv2.VideoCapture(0)
def lerImagem():
    caminhos = [os.path.join('fotos1',img) for img in os.listdir('fotos1')]
    
    faces1=[]
    ids = []
    
    for caminhosFaces in caminhos:
        imagemRosto = cv2.cvtColor(cv2.imread(caminhosFaces),cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhosFaces)[-1].split('.')[1])
        #print(id)
        #pessoa.1.1
        ids.append(id)
        hog=cv2.HOGDescriptor()
        h=hog.compute(imagemRosto)
        # h 907200
        faces1.append(h)
       # train_data = np.float32(faces).reshape(-1,len(faces[0]))
        #cv2.imshow("faces",imagemRosto)
        #cv2.waitKey(10)
        print('h ->  ',len(h))
       # cv2.imshow('frame',h)
    return np.array(ids),faces1




ids,faces1 = lerImagem()
#print(faces1)
#da uns print em cada variavel para ver o funcionamento
#aumentar a base de dados
#svm
# simular variavel de iluminação 

facesTreinamento=faces1[0:10]+faces1[15:25]
idsTreinamento=np.concatenate((ids[0:10],ids[15:25]))

facesTeste=faces1[10:15]+faces1[25:30]
idsTeste=np.concatenate((ids[10:15],ids[25:30]))

print("treinamento das imagens... ")

train_data = np.array(facesTreinamento).reshape(-1,len(facesTreinamento[0]))
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.train(np.array(train_data),cv2.ml.ROW_SAMPLE,np.array(idsTreinamento))
#svm.save('svm.yml')
#svm.setKernel(cv2.ml.SVM_LINEAR)
#svm.setType(cv2.ml.SVM_C_SVC)
#svm.setC(2.67)           # If necessary, we can change the C parameter of the classifier
#svm.setGamma(5.383)
#svm.train(train_data, cv2.ml.ROW_SAMPLE, ids)

teste_data = np.array(facesTeste).reshape(-1,len(facesTeste[0]))
teste=svm.predict(teste_data)
predictedIds=teste[1]
#print(predictedIds)
hog=cv2.HOGDescriptor()
faceVideo=[]
while(True):
    conectado,imagem = camera.read()
    
    if conectado == True:
        
        imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
        imagemCinza=imagemCinza / 2
        #reliza um cast 
        imagemCinza = imagemCinza.astype(np.uint8)
        
        retangulos = detectarFace.detectMultiScale(imagemCinza,
                                                 scaleFactor=1.5,
                                                 minSize=(30,30))
        if len(retangulos) > 0:
            faceDetectada = non_max_suppression(retangulos, probs=None, overlapThresh=0.6)

            #for(x,y,l,a) in faceDetectada:
            (x,y,l,a)=faceDetectada[0]
            imagemDaFace = cv2.resize(imagemCinza[y:y + a, x:x + l],(largura,altura))
            cv2.rectangle(imagem,(x,y),(x + l, y + a), (0,0,255),2)

            h=hog.compute(imagemDaFace)
            #faceVideo.append(h)
            #variante a escala
            verifica,id = svm.predict(np.array([h]))
            #print(id)
            nome = " "
            if id[0] == 1:
                nome = 'andre'
            elif id[0] == 2:
                nome = 'marcelo'
            cv2.putText(imagem,nome,(x,y +(a+30)),fonte,2, (0,0,255))
            cv2.putText(imagem,str(verifica),(x,y+(a+50)),fonte,1,(0,0,255))
        cv2.imshow("face ",imagem)
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()