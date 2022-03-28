
from cProfile import label
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
# load data
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Project\data.csv")
pd.options.display
df.head()
df.shape
#cdf = df['kathmandu.1','lalitpur.1','bhaktapur.1']
df1 = df.iloc[1:, 2:]
df1.head()

scaler = StandardScaler()
X = scaler.fit_transform(df1)
X
Kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(X)
labels = Kmeans.predict(X)
labels

np.unique(labels, return_counts=True)
df1['cluster'] = labels
df1.groupby('cluster').mean()
k_range = (2, 11)
sil_score = []
twss = []
for k in k_range:
    cluster = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster.fit(X)
    label = cluster.predict(X)
    ss = silhouette_score(X, label)
    sil_score.append(ss)
    twss.append(cluster.inertia_)
    


plt.plot(k_range, twss, "ro-")
plt.ylabel("twss")
plt.xlabel("number of clusters")
plt.plot(k_range, sil_score, "ro-")
plt.ylabel("silhouette score")
plt.xlabel("number of clusters")
T = PCA(n_components=2).fit_transform(X)
df2 = pd.DataFrame(T, columns=['PC1', 'PC2'])
plt.figure(figsize=(7, 5))
plt.scatter(df2.PC1, df2.PC2)

Kmeans=KMeans(n_clusters=4, n_init=20,random_state=0).fit(X)
labels=Kmeans.predict(X)
labels
df2['cluster']=labels
plt.figure(figsize=(10,10))
plt.scatter(df2.PC1,df2.PC2,c=df2['cluster'],s=50,cmap='rainbow')
df1['cluster']=labels
df1.groupby('cluster').mean()

pickle.dump(df1['cluster'], open('model.pkl', 'wb'))
