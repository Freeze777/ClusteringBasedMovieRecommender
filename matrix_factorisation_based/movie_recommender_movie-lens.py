
# coding: utf-8

# In[5]:

import numpy as np
import scipy.optimize as op


# In[67]:

n_movies=200
n_users=200
n_features=19 # comedy,action,romance etc. applies to both movie and user


# In[68]:

# movies x user rating matrix where ratings_matrix are from 1 to 10
#each column is a single users rating of all movies
# 0 value if the user has not rated the movie
ratings_matrix =np.zeros(shape=(n_movies,n_users))


# In[69]:

with open("ml-100k/ua.base") as f:
    for line in f:
        l=line.split("\t")
        user_id=int(l[0])
        movie_id=int(l[1])
        rating=int(l[2])
        if (movie_id <= n_movies) and (user_id <= n_users):
            ratings_matrix[movie_id-1][user_id-1]=rating
        



# In[70]:

did_rate=(ratings_matrix!=0)*1


# In[9]:

def normalize_ratings_matrix(ratings_matrix, did_rate):
    num_movies = ratings_matrix.shape[0]
    
    ratings_matrix_mean = np.zeros(shape = (num_movies, 1))
    ratings_matrix_norm = np.zeros(shape = ratings_matrix.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = np.where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_matrix_mean[i] = np.mean(ratings_matrix[i, idx])
        ratings_matrix_norm[i, idx] = ratings_matrix[i, idx] - ratings_matrix_mean[i]
    
    return ratings_matrix_norm, ratings_matrix_mean


# In[71]:

ratings_matrix, ratings_matrix_mean = normalize_ratings_matrix(ratings_matrix, did_rate)


# In[11]:


def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta


# In[12]:


def calculate_cost(X_and_theta, ratings_matrix, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = np.sum( (X.dot( theta.T ) * did_rate - ratings_matrix) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (np.sum( theta**2 ) + np.sum(X**2))
	return cost + regularization


# In[13]:


def calculate_gradient(X_and_theta, ratings_matrix, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings_matrix
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# In[72]:

reg_param = 0
movie_features = np.random.randn( n_movies, n_features )
user_prefs = np.random.randn( n_users, n_features )
initial_X_and_theta = np.r_[movie_features.T.flatten(), user_prefs.T.flatten()]

# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = op.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,
                                               args=(ratings_matrix, did_rate, n_users, n_movies, n_features, reg_param), 
                                               disp=True, full_output=True ) 


# In[ ]:

#np.save("minimized_cost_and_optimal_params.npy", minimized_cost_and_optimal_params)
cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, n_users, n_movies, n_features)
all_predictions = movie_features.dot( user_prefs.T )


# In[44]:

user_predictions = all_predictions[:,4:5]+ ratings_matrix_mean
print user_predictions[16]


# In[ ]:

test_ratings_matrix =np.zeros(shape=(n_movies,n_users))
with open("ml-100k/ua.test") as f:
    for line in f:
        l=line.split("\t")
        user_id=int(l[0])
        movie_id=int(l[1])
        rating=int(l[2])
        if (movie_id <= n_movies) and (user_id <= n_users):
            test_ratings_matrix[movie_id-1][user_id-1]=rating


# In[ ]:

predicted_unwatched=all_predictions*(1-did_rate)


# In[60]:

mse=0
for user in range(n_users):
    user_predictions = predicted_unwatched[:,user:user+1]+ ratings_matrix_mean
    user_actual=test_ratings_matrix[:,user:user+1]
    error=user_predictions-user_actual
    mse+=sum(error**2)

rmse=np.sqrt(mse/(n_users*n_movies))
print rmse


# In[ ]:



