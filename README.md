# Fake-News
**Objective:** 
We live in a crazy world with a lot of propaganda especially with the use of social media. I enjoyed doing this project because I got to use different methods and classifier such as Naives Bayes,
Decision Tree Classifier, logistics Regression, Multinomial NB, and confusion matrix.
#### Initial Steps
1.Imported Pandas and read from a csv file. The dataset was pretty large so I had to limit it to 10,000 rows. 
2. Dropped a column from the dataset that was not important
3. Used vectorization to convert words into numbs. 
4. Dropped stop words that commonly hold no value when processing data.
5. Split the data  into 70% training and 30% testing

### Decision Tree Classifier:
I trained the model and created my prediction using accuracy, f1, recall and precision call to determine my prediction results.
![image](https://user-images.githubusercontent.com/74514654/164131894-0621c84f-9e47-4c2b-b510-89ae895b9f09.png)

### MultinomialNB 
-Natural language processing algorithm that is based on text frequency.Multinomial regression also uses the maximum likelihood estimation to evaluate 
the probability.
![image](https://user-images.githubusercontent.com/74514654/164132040-f986f7a9-b955-43ba-97e1-0e65401956c8.png)

### K-folds classification
I split my dataset into 10 data samples and evaluated the accuracy through a limited data sample.
![image](https://user-images.githubusercontent.com/74514654/164132121-e8f30bff-6d84-4f17-8886-4f4ebfb56ff4.png)

### Random Tree classifier
Multitudes of trees are created and the most most frequent tree output is used for overall classification. But I also did tests on all features as well.
![image](https://user-images.githubusercontent.com/74514654/164132356-69920b19-9824-4184-8916-a012b18cb3a3.png)


(Although the methods of the Random Tree Classifier are meant to provide an overall better accuracy, Logistic Regression ended up with the most accurate output.
In the beginning of this project I limited my dataset to 10,000 and 70% was used for training and 30% was used for testing.
![image](https://user-images.githubusercontent.com/74514654/164132277-eb5c81ea-96c2-4b2e-b47a-55e26f9f2066.png)

### Confusion Matrix
-I used the Confusion matrix with Logistic regression because it had the highest accuracy to classify how much of the data used in testing was real or fake.
I think logistic Regression gave me the overall best accuracy because it works best with binary inputs. My data set was classified as 1=fake/unreliable 
and 0=Real/reliable.


![image](https://user-images.githubusercontent.com/74514654/164132525-c7605cf2-8f16-4b5a-b9bc-07c05e12ad1c.png)

