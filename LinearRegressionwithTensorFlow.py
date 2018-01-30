

# # Linear Regression
# 
# Defining a linear regression in simple terms, its the approximation of a linear model used to describe the relationship betweeen two or more variables. In a simple linear regression there are two variables, the dependent variable, which can be seen as the "state" or "final goal" we study and try to predict, and the independent variables, also known as explanatory variables, which can be seen as the "causes" of the "states". 
# 
# When more than one independent variable is present the process is called multiple linear regression. When multiple dependent variables are predicted the process is known as multivariate linear regression.
# 
# The very known equation of a simple linear model is
# 
# $$Y = a X + b $$
# 
# Where Y is the dependent variable and X is the independent variable, and <b>a</b> and <b>b</b> being the parameters we adjust. <b> a </b> is known as "slope" or "gradient" and <b> b </b> as "intercept". You can interpret this equation as Y being a function of X, or Y being dependent of X.
# 
# If you plot the model, you will see it is a line, and by adjusting the "slope" parameter you will change the angle between the line and the independent variable axis, and the "intercept parameter" will affect where it crosses the dependent variable axis.
# 
# Let's first import packages:


import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 6)


# Let's define an independent variable:



X = np.arange(0.0, 5.0, 0.1)





##You can adjust the slope and intercept to verify the changes in the graph
a=1
b=0

Y= a*X + b 

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# OK... but how can we see this concept of linear relations with a more meaningful point of view?
# 
# Simple linear relations were used to try to describe and quantify many observable physical phenomena, the easiest to understand are speed and distance traveled:


# 
# Distance Traveled = Speed * Time + Initial Distance
# 
# Speed = Acceleration * Time + Initial Speed


# They are also used to describe properties of different materials:



# Force = Deformation * Stiffness 
# 
# Heat Transfered = Temperature Difference * Thermal Conductivity 
# 
# Electrical Tension (Voltage) = Electrical Current * Resistance
# 
# Mass =  Volume * Density


# When we perform an experiment and gather the data, or if we already have a dataset and we want to perform a linear regression, what we will do is adjust a simple linear model to the dataset, we adjust the "slope" and "intercept" parameters to the data the best way possible, because the closer the model comes to describing each ocurrence, the better it will be at representing them.
# 
# So how is this "regression" performed?

# ---------------


# # Linear Regression with TensorFlow
# A simple example of a linear function can help us understand the basic mechanism behind TensorFlow. 
# 
# For the first part we will generate random data points and define a linear relation, we'll use TensorFlow to adjust and get the right parameters.
# 


x_data = np.random.rand(100).astype(np.float32)


# The equation for the model used in this example is :
# 
# $$Y = 3 X + 2 $$
# 
# 
# Nothing special about this equation, it is just a model that we use to generate our data points. In fact, you can change the parameters to whatever you want, as you will do later. We add some gaussian noise to the points to make it a bit more interesting.



y_data = x_data * 3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)


# Here is a sample of data:

:

zip(x_data,y_data) [0:5]


# First, we initialize the variables __a__ and __b__, with any random guess, and then we define the linear function:



a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b


# In a linear regression, we minimize the squared error of the equation that we want to adjust minus the target values (the data that we have), so we define the equation to be minimized as loss.
# 
# To find Loss's value, we use __tf.reduce_mean()__. This function finds the mean of a multidimensional tensor, and the result can have a diferent dimension.



loss = tf.reduce_mean(tf.square(y - y_data))


# Then, we define the optimizer method. Here we will use a simple gradient descent with a learning rate of 0.5: <br/>  
# Now we will define the training method of our graph, what method we will use for minimize the loss? We will use the tf.train.GradientDescentOptimizer.  
# .minimize()__ will minimize the error function of our optimizer, resulting in a better model.



optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# Don't forget to initialize the variables before executing a graph:



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Now we are ready to start the optimization and run the graph:



train_data = []
for step in range(100):
    evals = sess.run([train,a,b])[1:]
    if step % 5 == 0:
        print(step, evals)
        train_data.append(evals)




converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')


green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()


# 
