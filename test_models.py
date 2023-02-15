import numpy as np
from sklearn.datasets import make_regression
from joblib import load 
import pandas as pd
from os.path import join
import time 
import onnxruntime as rt


def time_it(model_name, model, X,iters=10,onnx=False):
    res = []
    if not onnx:
        for i in range(iters):
            tic = time.time()
            pred = model.predict(X)
            toc = time.time()
            res.append((toc - tic))
            
    return pd.DataFrame({'model':model_name, 'time':res})


def time_it_onnx(sess,label_name, input_name, model_name,iter=10):
    res = []
    for i in range(iter):
        tic = time.time()
        predictions = sess.run([label_name],
                          {input_name:  X.astype(np.float32)})
        toc = time.time()
        res.append((toc - tic))
    return pd.DataFrame({'model':model_name, 'time':res})
    

models_path = 'models'


df_list = []

print (rt.get_device())
print (rt.get_available_providers())
#print (rt.get_all_providers())


# Generación del dataset
n_samples = 1000000
n_features = 40
X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=0)


lr = load(join(models_path,'lr.joblib'))
rf_model = load(join(models_path,'rf.joblib'))


lr_results = time_it('lr', lr, X,iters=10)
df_list.append(lr_results)
lr_results

rf_results = time_it('rf_model', rf_model, X,iters=10)
df_list.append(rf_results)
rf_results

sess = rt.InferenceSession(join(models_path,'catb.onnx'),providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
model_name = 'cat_onnx'

cat_results_onnx = time_it_onnx(sess,label_name, input_name, model_name,iter=10)
df_list.append(cat_results_onnx)
cat_results_onnx

sess = rt.InferenceSession(join(models_path,'rf.onnx'))

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

model_name = 'rf_onnx'

rf_results_onnx = time_it_onnx(sess,label_name, input_name, model_name,iter=10)
df_list.append(rf_results_onnx)

rf_results_onnx

sess = rt.InferenceSession(join(models_path,'lr.onnx'))


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

model_name = 'lr_onnx'

lr_results = time_it_onnx(sess,label_name, input_name, model_name,iter=10)
df_list.append(lr_results)
lr_results

metrics = pd.concat(df_list)

metrics.groupby('model').mean()


print (metrics)
