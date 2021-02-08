import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import skimage.color
import skimage.feature
import skimage.io
import skimage
import sklearn
from sklearn.svm import SVC
import scipy
import pickle



# class rgb2gray_transform(BaseEstimator,TransformerMixin):
#     import skimage.color
#     def __init__(self):
#         pass
    
#     def fit(self,X,y=None):
#         return self
    
#     def transform(self,X,y=None):
#         return np.array([skimage.color.rgb2gray(x) for x in X])

# class hogtransformer(BaseEstimator,TransformerMixin):
#     import skimage.feature
#     def __init__(self,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3),):
#         self.orientations = orientations
#         self.pixels_per_cell = pixels_per_cell
#         self.cells_per_block = cells_per_block
        
        
#     def fit(self,X,y=None):
#         return self
    
#     def transform(self,X,y=None):
#         def local_hog(img):
#             hog_features= skimage.feature.hog(img,orientations=self.orientations,
#                                 pixels_per_cell=self.pixels_per_cell,
#                                 cells_per_block=self.cells_per_block)
            
#             return hog_features
        
#         hfeatures = np.array([local_hog(x) for x in X])
#         return hfeatures



def top_five_results(model,le,image_path):
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
