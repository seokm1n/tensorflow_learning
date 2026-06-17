import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications import vgg16  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from sklearn.metrics import accuracy_score # 예측 성능 평가

newmodel = load_model("/home/sm/tf_env/20260617/covid19_best.keras")  # 앞선 훈련 완료 모델 로드

pred_list = [] # 예측 결과 저장 리스트

# test 이미지 데이터 로딩 및 예측 수행
def predict_vgg16_newmodel(newmodel, filename): # 파일 별 예측 결과 저장함수
    img = image.load_img(filename, target_size=(224, 224)) # 파일 이미지 로드
    img_arr = image.img_to_array(img) # image 데이터 넘파일 배열로 변환

    image_reshape = img_arr.reshape((1, 224, 224, 3))
    image_input = vgg16.preprocess_input(image_reshape) # vgg16 모델입력전처리# {'Covid': 0, 'Normal': 1}

    pred = newmodel.predict(image_input, batch_size=1) # 해당 이미지파일예측#print(pred)
    class_list = ['covid19', 'normal']
    print('pred result : ', class_list[np.argmax(pred)]) # 예측 최대 추정치 인덱스추출
    pred_list.append( class_list[np.argmax(pred)] ) # 예측 결과 list에 저장

test_dir = "/home/sm/tf_env/20260617/Covid19-dataset/test/Covid/"
filenamelist = os.listdir(test_dir) # 디렉토리 내부의 모든 파일 정보 리스트 반환
print(filenamelist)

file_totalinfo = []

for file in filenamelist:
    file_totalinfo.append(test_dir+file)

print(file_totalinfo) # 파일 경로 + 파일name 정보리스트

for imagefile in file_totalinfo:
    predict_vgg16_newmodel(newmodel, imagefile) # 예측 함수 호출

class_name = 'covid19' # covid test image로 예측수행했음으로 분류이름을 ‘covid19’로사용

df = pd.DataFrame({'True_Data':[class_name]*len(file_totalinfo), 
                   'Pred_Data':pred_list, 'filename':filenamelist})
print(df)
print('accuracy : %.3f' %accuracy_score(df['True_Data'], df['Pred_Data']))