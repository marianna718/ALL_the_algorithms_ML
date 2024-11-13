import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from numpy import loadtxt

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():

    file = open('./data/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./data/small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('./data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)


# GRADED FUNCTION: cofi_cost_func
# UNQ_C1

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    reg_x = 0
    reg_w =0
    Total = 0.0
    for i in range(nm):
        for j in range(nu):
            dist = ((np.dot(W[j,:],X[i,:].T) + b[:,j]) - Y[i,j])**2
#             print("dist: ",dist)
            Total = Total + R[i,j] * dist[0]
#             print("Total: ",Total)
#             print("Rij: ",R[i,j])
#             print(i,j)
#             print("Wij shape: ",W[j,:].shape)
            
            
        for k in range(X.shape[1]):
            reg_x += X[i,k]**2
    for m in range(nu):
        for l in range(X.shape[1]):
                reg_w += W[m,l]**2
#     #######Regularization#########
    
    J = Total/2
    J = J + (reg_x * (lambda_/2)) + (reg_w *(lambda_/2))
    print((J))
            


    return J

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def prediction_maker_tr(X,W,b,Ynorm,R, optimizer,lambda_ = 1):
    iterations = 200
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        # Log periodically.
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
        


def main():
    print("This is the main function.")

    #Load data
    X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
    Y, R = load_ratings_small()

    # print("Y", Y.shape, "R", R.shape)
    # print("X", X.shape)
    # print("W", W.shape)
    # print("b", b.shape)
    # print("num_features", num_features)
    # print("num_movies",   num_movies)
    # print("num_users",    num_users)

    # From the rating Y, we can compute statistics like average rating for a movie

    print("\n\n LETS CHECK OUT HOW THE COST FUNCTION WORKS, please feel in next informations \n")
    # num_users_r = 4
    # num_movies_r = 5
    # num_features_r = 3
    # lam_ = 1.5

    num_users_r = int(input("Please enter the number of users you want to consider out of 443:(Recomended 4)  "))
    num_movies_r = int(input("Please enter the number of movies you want to consider out of 4778:(Recomended 5)  "))
    num_features_r = int(input("Please enter the number of feautures you want to consider out of 10:(Recomended 3)  "))
    lam_ = float(input("Please enter the regularization term(Recomended 1.5) "))
    X_r = X[:num_movies_r, :num_features_r]
    W_r = W[:num_users_r,  :num_features_r]
    b_r = b[0, :num_users_r].reshape(1,-1)
    Y_r = Y[:num_movies_r, :num_users_r]
    R_r = R[:num_movies_r, :num_users_r]

    # Evaluate cost function

    J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, lam_)
    print(f"Cost: {J:0.2f}")


    movieList, movieList_df = load_Movie_List_pd()

    my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Now we will gather the ratings of the user and add them to the rating dataset as result of other user

    print(" \n\n NOW!, Lets find out your oppinion about following movies \n to give you the precise predictions of some \n movies that you may like \n")
    
    print("Rate them from -0 to 5-, with 0.5 steps, \n if you haven't wached them yet rate them as -0- \n")

# un-coment this section if you dont want to give your rateings to the movies

    # my_ratings[2700] = 5   # Toy Story 3 (2010)
    # my_ratings[2609] = 2   # Persuasion (2007)
    # my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
    # my_ratings[246]  = 5   # Shrek (2001)
    # my_ratings[2716] = 3   # Inception
    # my_ratings[1150] = 5   # Incredibles, The (2004)
    # my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
    # my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    # my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
    # my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
    # my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
    # my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
    # my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    
    my_ratings[2700] = float(input("Toy Story 3 (2010)  -> ")) 
    my_ratings[2609] = float(input("Persuasion (2007)  -> " ))
    my_ratings[929]  = float(input("Lord of the Rings: The Return of the King  -> "))
    my_ratings[246]  = float(input("Shrek (2001)  -> "))
    my_ratings[2716] = float(input("Inception  -> "))
    my_ratings[1150] = float(input("Incredibles, The (2004)  -> "))
    my_ratings[382]  = float(input("Amelie (Fabuleux destin d'Amélie Poulain, Le)  -> "))
    my_ratings[366]  = float(input("Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)  -> "))
    my_ratings[622]  = float(input("Harry Potter and the Chamber of Secrets (2002)  -> "))
    my_ratings[988]  = float(input("Eternal Sunshine of the Spotless Mind (2004)  -> "))
    my_ratings[2925] = float(input("Louis Theroux: Law & Disorder (2008)  -> "))
    my_ratings[2937] = float(input("Nothing to Declare (Rien à déclarer)  -> "))
    my_ratings[793]  = float(input("Pirates of the Caribbean: The Curse of the Black Pearl (2003)  -> "))


    my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

    print('\nNew user ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0 :
            print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}')


    # Reload ratings
    Y, R = load_ratings_small()

    # Add new user ratings to Y 
    Y = np.c_[my_ratings, Y]

    # Add new user indicator matrix to R
    R = np.c_[(my_ratings != 0).astype(int), R]

    # Normalize the Dataset
    Ynorm, Ymean = normalizeRatings(Y, R)
# This on top is super helpfull for new movies and new users, to be short helps with cold starts to not be 0-fied ;)

    #  Useful Values
    num_movies, num_users = Y.shape
    num_features = 100

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    prediction_maker_tr(X,W,b,Ynorm,R, optimizer,lambda_ = 1)

    # Make a prediction using trained weights and biases
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    #restore the mean
    pm = p + Ymean

    my_predictions = pm[:,0]

    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')
    print("\n\n     WOW, YOU'LL DEFINITELY LOVE THESE MOVIES!   \n\n")
    for i in range(17):
        j = ix[i]
       
        if j not in my_rated:
            print(f'Predicting rating {my_predictions[j]:0.2f} for movie ---> {movieList[j]} <---')

    print('\n\n Original vs Predicted ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')


    print("The end of the main funcion.")




if __name__ == "__main__":
    main()
