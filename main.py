from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import skimage.color
import skimage.feature
import skimage.transform
import pickle
import skimage.io
import os
import scipy
import joblib
from models import rgb2gray_transform, hogtransformer

app = Flask(__name__)
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR,'static/models/')
UPLOAD_PATH =os.path.join(BASE_DIR,'static/upload/')

le = joblib.load(os.path.join(MODEL_PATH,'labelenco.joblib'))



@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'image_name' in request.files:
                
            print('you are in post')
            upload_file = request.files['image_name']
            filename = upload_file.filename
            filepath = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(filepath)
            print('name of the file =',filename)
            res = top_five_results(filepath)
            height_img = getheight(filepath)
            print(res,height_img)
            # except:
            #     print('Something went wrong')
            return render_template('now.html',fileupload=True,image_name=filename,
            results=res,h=height_img)

        else:
            file_url = request.form['img_url']
            print('the urls is ',file_url)
            img_arr = skimage.io.imread(file_url)
            filename = file_url.split('/')[-1]
            print('Filename is', filename)

            filepath = os.path.join(UPLOAD_PATH,filename)
            skimage.io.imsave(filepath,img_arr)
            
            res = top_five_results(filepath)
            height_img = getheight(filepath)
            print(res,height_img)
            # save image
            return render_template('now.html',fileupload=True,
            image_name=filename,results=res,h=height_img)
        
    else:
        return render_template('now.html', fileupload=False)

# Machine Learning Model

def getheight(filepath):
    img = skimage.io.imread(filepath)
    height,width, _ = img.shape
    aspect = height / width
    w = 300
    h = w* aspect
    
    return h



def top_five_results(image_path):
    img_test= skimage.io.imread(image_path)
    # image size is 80 x 80
    img_resize = skimage.transform.resize(img_test,(80,80))
    # rescale into 255
    img_rescale = np.array(255*img_resize).astype(np.uint8)
    # machine leanring
    img_reshape = img_rescale.reshape(-1,80,80,3)
    pred = model.predict(img_reshape)[0]
    val = le.inverse_transform(pred.flatten())
    # Descision Function
    distance = model.decision_function(img_reshape)[0]
    # top 5 prediction
    z = scipy.stats.zscore(distance)
    pvals = scipy.special.softmax(z)
    index = pvals.argsort()[-5:][::-1]
    top_class = le.classes_[index]
    score = np.round(pvals[index],2)
    prediction_dict ={}
    for i,j in zip(top_class,score):
        prediction_dict.update({i:j})

    return prediction_dict


if __name__ == '__main__':
    # with open(os.path.join(MODEL_PATH,'labelencoder.pickle'),'rb') as f:
    

    # with  as f:
    model = pickle.load(open(os.path.join(MODEL_PATH,'model_pipeline_svm.pickle'),'rb'))
    
    app.run()