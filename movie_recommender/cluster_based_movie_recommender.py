
# coding: utf-8

# In[15]:

from movielens import *
import sys
import time
import math
import re
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import mean_squared_error


# In[31]:

users=[]
movies=[]
ratings=[]
data_loader=Dataset()
data_loader.load_users("data/u.user",users)
data_loader.load_items("data/u.item",movies)
data_loader.load_ratings("data/u.base",ratings)


# In[32]:

movie_features=[]
for movie in movies:
    movie_features.append([movie.unknown,movie.action,movie.adventure,movie.animation,
                            movie.childrens,movie.comedy,movie.crime,movie.documentary,
                            movie.drama,movie.fantasy,movie.film_noir,movie.horror,
                            movie.musical,movie.mystery,movie.romance,movie.sci_fi,
                            movie.thriller,movie.war,movie.western])


# In[35]:

num_users=len(users)
num_movies=len(movies)
num_features=len(movie_features[0])
print num_users,num_movies,num_features


# In[53]:

clusterer=KMeans(n_clusters=num_features)
#clusterer=MiniBatchKMeans(n_clusters=num_features, batch_size=500)
cluster_label_of_movies=clusterer.fit_predict(movie_features)


# In[74]:

ratings_matrix=np.zeros(shape=(num_users,num_movies))

for rating in ratings:
    ratings_matrix[rating.user_id-1][rating.item_id-1]=rating.rating
    
print ratings_matrix


# In[55]:

users_cluster_avg=[]
for user in users:
    cluster_avg_rating=np.zeros(num_features)
    cluster_of_rated_movies=[[] for i in range(num_features)]
    for movie in movies:
        if ratings_matrix[user.id-1][movie.id-1]!=0:
            cluster_of_rated_movies[cluster_label_of_movies[movie.id-1]-1].append(ratings_matrix[user.id-1][movie.id-1])
    for i in range(num_features):
        if len(cluster_of_rated_movies[i])!=0:
            cluster_avg_rating[i]=np.mean(cluster_of_rated_movies[i])
        else:
            cluster_avg_rating[i]=0 #user havent seen any movie in this cluster
    users_cluster_avg.append(cluster_avg_rating)
        
users_cluster_avg=np.array(users_cluster_avg)
print users_cluster_avg


# In[56]:

for user in users:
    cluster_avg_rating=users_cluster_avg[user.id-1]
    sum_of_avg_of_rated_clusters=sum(r for r in cluster_avg_rating if r>0)
    num_of_rated_clusters=sum(r>0 for r in cluster_avg_rating)
    user.avg_r=sum_of_avg_of_rated_clusters/num_of_rated_clusters
    


# In[57]:

def pearsons_correlation_coefficient(user1,user2):
    user1_cluster_avg_list=users_cluster_avg[user1-1]
    user2_cluster_avg_list=users_cluster_avg[user2-1]
    numerator=0.0
    denominator1=0.0
    denominator2=0.0
    
    for u1,u2 in zip(user1_cluster_avg_list,user2_cluster_avg_list):
        if u1>0 and u2>0:
            numerator+=((u1-users[user1-1].avg_r)*(u2-users[user2-1].avg_r))
        if u1>0:
            denominator1+=(u1-users[user1-1].avg_r)**2
        if u2>0:
            denominator2+=(u2-users[user2-1].avg_r)**2
            
    denominator=(denominator1*denominator2)**0.5
    if denominator==0:
        return 0
    else:
        return numerator/denominator


# In[58]:

user_similarity_matrix=np.zeros(shape=(num_users,num_users))
for i in range(num_users):
    for j in range(i):
            user_similarity_matrix[i][j]=pearsons_correlation_coefficient(i+1,j+1)
            user_similarity_matrix[j][i]=user_similarity_matrix[i][j]

print user_similarity_matrix


# In[65]:

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return

def load_object(filename):
    with open(filename, 'rb') as input:
        obj=pickle.load(input)
    return obj

save_object(user_similarity_matrix,"user_similarity_matrix.pkl")


# In[76]:

def predict_user_cluster_rating(user_id, clust_id, top_n):
    similar_users=[]
    for i in range(num_users):
        if user_id-1!=i:
            similar_users.append(user_similarity_matrix[user_id-1][i])
            
    users_cluster_avg_cpy=np.copy(users_cluster_avg)
    users_cluster_avg_cpy=np.delete(users_cluster_avg_cpy, user_id-1, 0)
    sorted_similar_users=[x for (y,x) in sorted(zip(similar_users,users_cluster_avg_cpy), key=lambda pair: pair[0], reverse=True)]
    #sorted_similar_users=sorted(zip(similar_users,users_cluster_avg_cpy),reverse=True)[:top_n]
    
    s=0
    c=0
    for i in range(top_n):
        if sorted_similar_users[i][clust_id-1]!=0:
            s+=sorted_similar_users[i][clust_id-1]
            c+=1
    for i in range(top_n//3):
        if sorted_similar_users[-(i+1)][clust_id-1]!=0:
            s+=(6-sorted_similar_users[-(i+1)][clust_id-1])
            c+=1
   
    pred = users[user_id-1].avg_r
    if c != 0:
        pred=s/float(c)
        
    if pred < 1.0:
        return 1.0
    elif pred > 5.0:
        return 5.0
    else:
        return pred


# In[77]:


users_cluster_avg_cpy = np.copy(users_cluster_avg)
for i in range(num_users):
    for j in range(num_features):
        if users_cluster_avg_cpy[i][j] == 0:
            users_cluster_avg_cpy[i][j] = predict_user_cluster_rating(i+1, j+1, 150)


print users_cluster_avg_cpy

save_object(users_cluster_avg_cpy,"users_cluster_avg_cpy.pkl")


# In[80]:

rating_test = []
data_loader.load_ratings("data/u.test", rating_test)
test = np.zeros((num_users, num_movies))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating
    
y_true = []
y_pred = []

for i in range(0, num_users):
    for j in range(0, num_movies):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(users_cluster_avg_cpy[i][cluster_label_of_movies[j]-1])


print "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred)


# In[ ]:



