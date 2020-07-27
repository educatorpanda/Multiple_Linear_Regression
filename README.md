## Python Code for Multiple Linear Regression

* **Data Preparation**

We will use a real life data which is built for multiple linear regression and multivariate analysis, known as the Fish Market Dataset that contains information about common fish species in market sales. The dataset includes the fish species, weight, length (of 3 types), height, and width. Our main objective is to predict the dependent variable **weight** using the independent variables **length (of 3 types)**, **height**, and **width**
Its a .csv file and you can download this dataset from [here](https://www.kaggle.com/aungpyaeap/fish-market/data)

Let us first import all the important libraries. 
 
{% highlight python linenos %}
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
{% endhighlight %}

Now let us load the data (Fish.csv) in the 'df' variable using ***pandas*** library and preview the first 5 lines of the loaded data

{% highlight python linenos %}
df = pd.read_csv("Fish.csv") 
df = df.sample(frac=1).reset_index(drop=True)
df.head()
{% endhighlight %}

In **line 2** of the above code snippet, we are randomly shuffling the dataframe in-place and resetting the index. Here, specifying **drop=True** prevents ***.reset_index*** from creating a column containing the old index entries.

![Data2](/assets/img/out1.png)

So, as you can see, the column **"species"** includes different types of fish species available based on its different attributes such as **weight**, **length**, **height** and **width**. Now let us print and check the number of samples (data points) available for our model.

{% highlight python linenos %}
print(df.shape)
{% endhighlight %}

```python
out: (159,7)
```

In a dataset, a training set is implemented to build up a model, while a test (or validation) set is to validate the model built. So, we use the training data to fit the model and testing data to test it. Therefore, we will be splitting 80% and 20% of our dataset into train and test data respectively. This will be accomplished using **train_test_split** function of ***sklearn*** library. 

{% highlight python linenos %}
train, test = train_test_split(df, test_size=0.2)
{% endhighlight %}

Let us verify the same by printing the rows and columns information of train and test samples

{% highlight python linenos %}
print(train.shape, test.shape)
{% endhighlight %}

```python
out: (127,7), (32,7)
```
One final thing we need to do is to separate the independent and dependent variables from the dataset and store it as a numpy array to perform matrix calculations!

{% highlight python linenos %}
Xtrain, Ytrain = train.loc[:,'Length1':'Width'].to_numpy(), train.loc[:,'Weight'].to_numpy() 
Xtest, Ytest = test.loc[:,'Length1':'Width'].to_numpy(), test.loc[:,'Weight'].to_numpy()
{% endhighlight %}

Now we are all set to perform regression analysis. For that let us define a function called **MLR** as follows:

{% highlight python linenos %}
def MLR(X,Y):
    t1 = np.linalg.inv(np.dot(X.transpose(),X))
    t2 = np.dot(X.transpose(),Y)
    w = np.dot(t1,t2)
    return w
{% endhighlight %}

This function takes 2 inputs (**X** and **Y**) and returns the optimum value of weights and bias using this formula 

![wformula](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20w%3D%5Cleft%20%28%20X%5E%7B%5Ctop%7DX%20%5Cright%20%29%5E%7B-1%7DX%5E%7B%5Ctop%7DY){: .mx-auto.d-block :}

We will also define another function called **testMLR** which takes 2 inputs (**X** and **w**) returns the predicted output **y**

{% highlight python linenos %}
def testMLR(X,w):
    y = np.dot(X,w)
    return y
{% endhighlight %}

Now let us call both of these functions and calculate **w** and **y**

{% highlight python linenos %}
w = MLR(Xtrain,Ytrain)
y = testMLR(Xtest,w)
{% endhighlight %}

Finally, let us compare our original output **Ytest** with the predicted output **y** by plotting them using ***matplotlib.pyplot*** library

{% highlight python linenos %}
plt.plot(y, "-b", label="y")
plt.plot(Ytest, "-r", label="Ytest[]")
plt.legend(loc="upper right")
plt.show()
{% endhighlight %}

This is what we get after executing the above code:

![Data2](/assets/img/out2.png){: .mx-auto.d-block :}
