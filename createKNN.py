import pandas as pd


df = pd.read_csv('assets/sizes.csv')

df['Gender'] = df['Gender'].replace({'Male': [0], 'Female':[1], 'Child':[2]})
df['Age-Group'] = df['Age-Group'].replace({'Adult':0, 'Child':1})
print(df)
train = df[['Gender','Age-Group','Chest','Waist','Hip','Neck-int','Thigh']]
labels = df['Size']

from sklearn.neighbors import KNeighborsClassifier as KNN 
knn1 = KNN(n_neighbors = 1) 
  
# train model 
knn1.fit(train, labels) 


import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(knn1, 'assets/knn1.pkl') 

