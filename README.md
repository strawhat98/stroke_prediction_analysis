# Stroke Prediciton Analysis

Effects of a stroke depend on several factors, including daily routine of a person, his or her environmental factor, stress level which depends upon professional and personal life, Eating habits and many other catalysts. Medical science says stroke is not only stress induced bodily behaviour but also can genetically propagate maong generations of people in their family. So, to determine how stroke can be predicted based on typical characteristics of human life, this prediction analysis has been completed using different predictive modelling(Random Classifier Model, Logistic Regression Model, Naive-Bayes, KNN- Classifier,etc.) to generate efficient outcome for future prupose.


# -:OBJECTIVE:-
   • The Stroke Prediction Model has been designed to predict if a person with standard health conditions can be a victim of stroke in his or her future or not.
      To create a model that will employ some machine learning algorithms provided by the libraries of programming language ‘python’. 
      
   • To choose a target function and a suitable experience for creating the model.
      
   • To check the skewness of data and reduce redundancy, NaN value removal, Univariate and bivariate analysis to manage data consistency.
      
   • To apply different model upon the data for gathering precision and recall value.Feature selection using rapper method on data and through ensemble learning calculating accuracy on prediction.
    
# -:SCOPE:-
   • This model will help predict health condition of person with standard life style parameters .
      
   •  The model, if used, will help avoid the trouble of future risk of a person for stroke and what kind of lifestyle a person should lead.
      
   • The model will learn from each experience and increase its accuracy on its own without any alteration or intervention by a programmer.
   
## Prerequisites:
- Dataset
- Jupyter Notebook
- pip (python 3)

## pip libraries to use:
1. seaborn
2. pandas
3. Matplotlib
4. Numpy
5. Scikit Learn

# Dataset Attribute Information (Source - Kaggle platform):
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_job", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not.

### Sample Dateset:-

![image](https://user-images.githubusercontent.com/39863708/122547801-2fa06780-d04e-11eb-98a7-21ff2b023fbf.png)

# WorkFlow:-

![image](https://user-images.githubusercontent.com/39863708/122674737-1507f300-d1f4-11eb-90c8-145686a48452.png)


# Exploratory data analysis:


a) at first for each attribute present in the dataset, unique values are checked upon the total value present actually.


![image](https://user-images.githubusercontent.com/39863708/122547884-48a91880-d04e-11eb-86c7-1f7ba5ba904b.png)


![image](https://user-images.githubusercontent.com/39863708/122547891-4a72dc00-d04e-11eb-8632-296f782033a9.png)
![image](https://user-images.githubusercontent.com/39863708/122547895-4c3c9f80-d04e-11eb-9082-3e8bbdb8536d.png)

![image](https://user-images.githubusercontent.com/39863708/122547906-4e9ef980-d04e-11eb-9a94-e67e20480127.png)
![image](https://user-images.githubusercontent.com/39863708/122547912-5068bd00-d04e-11eb-9b10-61f9a8db9396.png)



b) using density plot or displot() checking skewness of data considering the data is sorted and as the ‘bmi’ value has less more unique values than other datasets, plotting the ‘bmi’ distplot elaborately.

![image](https://user-images.githubusercontent.com/39863708/122547939-59598e80-d04e-11eb-8919-9c711e3ba24a.png)

Skewness:-
Skewness is the measurement of asymmetry or distortion of symmetry distribution. It measures deviation of the given distribution of a random variable from a symmetric distribution such as normal distribution. A normal distribution is without any skewness which means it is symmetrical on both side.

Skewness are two types, Positive skewness and Negative skewness.

This ‘bmi’ data is positively skewed as extreme data results are larger which brings the mean value of data grater which is larger than the median value.
Skewness of all the attributes inside the data,

![image](https://user-images.githubusercontent.com/39863708/122547960-64142380-d04e-11eb-9ab1-dadb560c1221.png)

only larger attributed data's will be checked for skewness.
(‘bmi’ , ‘average glucose level’)

Skewness Removal Technique:-

Transformation used to reduce a distribution that is symmetric or nearly symmetric so is often easier to handle and interpret skewer distribution.
To remove skewness here Square-root method is applied,
mainly existing data is square rooted multiple times until the density plot of the data becomes balanced.


![image](https://user-images.githubusercontent.com/39863708/122548014-72623f80-d04e-11eb-9275-307a8e431866.png)
![image](https://user-images.githubusercontent.com/39863708/122548023-74c49980-d04e-11eb-82c1-a9ecc637d533.png)


same way, average glucose level is levelled up.

![image](https://user-images.githubusercontent.com/39863708/122548062-80b05b80-d04e-11eb-8db5-0be2c958c61e.png)
![image](https://user-images.githubusercontent.com/39863708/122548077-8443e280-d04e-11eb-9f41-7ff3de4da9a6.png)
![image](https://user-images.githubusercontent.com/39863708/122548088-89089680-d04e-11eb-819d-89efcb22792a.png)

3) Categorical to numerical value conversio:
![image](https://user-images.githubusercontent.com/39863708/122548131-958cef00-d04e-11eb-9b96-dbe2435b5411.png)
Attribute ever_married is of categorical data where Yes and No value is changed into 0 and 1 which is then concatenated to the actual data and the ever_married column is dropped from the actual dataset.

The column in which the data is stored is Yes which is changed to Marital status considering Yes value as 1 and No value as 0.
![image](https://user-images.githubusercontent.com/39863708/122548154-9b82d000-d04e-11eb-8636-aa3908e9bde7.png)
![image](https://user-images.githubusercontent.com/39863708/122548161-9e7dc080-d04e-11eb-9ca2-ae40542854c2.png)


Fit transform:-

    • Data standardization is the process of rescaling the attributes so that they have mean as 0 and variance as 1.
    • The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values.
    • In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently on each feature.
    
the same way as before gender ,work_type, residence type, smoking status all the categorical data values are transformed using fit transformation into numerical values column is changed into numerical values using fit transform method.

After fit transform all the categorical values are transformed,
![image](https://user-images.githubusercontent.com/39863708/122548206-accbdc80-d04e-11eb-936c-f9f2f83e98de.png)

4) Univariate Analysis using Violin plot:-

The purpose of univariate analysis is to understand the distribution of values for a single variable. 

![image](https://user-images.githubusercontent.com/39863708/122548222-b35a5400-d04e-11eb-8e39-d43212fa3091.png)

Univariate analysis using Countplot:-
![image](https://user-images.githubusercontent.com/39863708/122548255-bd7c5280-d04e-11eb-8000-c966f31601ab.png)
![image](https://user-images.githubusercontent.com/39863708/122548264-bfdeac80-d04e-11eb-82e6-d101404830c9.png)

Heat Map Analysis:-
![image](https://user-images.githubusercontent.com/39863708/122548283-c5d48d80-d04e-11eb-8b30-b9a6b1abbd77.png)
5) Bivariate analysis using count plot and scatter plot:
![image](https://user-images.githubusercontent.com/39863708/122548311-cff68c00-d04e-11eb-91a9-c4c75cbc5c8e.png)
![image](https://user-images.githubusercontent.com/39863708/122548317-d258e600-d04e-11eb-8ca2-3ef6c6e81e93.png)

![image](https://user-images.githubusercontent.com/39863708/122548331-d84ec700-d04e-11eb-87b4-bf93ae27c344.png)
![image](https://user-images.githubusercontent.com/39863708/122548353-e00e6b80-d04e-11eb-9042-68b84eb7fd6d.png)

![image](https://user-images.githubusercontent.com/39863708/122548364-e270c580-d04e-11eb-8fa4-b9084b2cc439.png)

![image](https://user-images.githubusercontent.com/39863708/122548373-e43a8900-d04e-11eb-8206-18ae1a746b17.png)
6) Feature Selection:-
![image](https://user-images.githubusercontent.com/39863708/122548404-eef51e00-d04e-11eb-897d-7f28903099d3.png)
![image](https://user-images.githubusercontent.com/39863708/122548409-f3213b80-d04e-11eb-8b9f-ad58acd911c5.png)
![image](https://user-images.githubusercontent.com/39863708/122548417-f6b4c280-d04e-11eb-9c39-9853a00e70cc.png)

7) Models on all features:-
Used models :- Logistic Regression Model, Decision Tree Classifier, KNN, naive_Bayes Classifier, SVM & Random Forest.

Best Model predicted for the upsampled data is,
                                    KNN model (recall is 85.50)

Worst Model Predicted is, 
                                   Naive-Bayes model(recall is 79.06)

SO KNN Model is shown here,
![image](https://user-images.githubusercontent.com/39863708/122548464-0502de80-d04f-11eb-98d7-2652495ab1b1.png)
![image](https://user-images.githubusercontent.com/39863708/122548471-07653880-d04f-11eb-8acd-26704b7773d3.png)

) backward Elimination:-
In backward elimination using different number of columns inside prediction model all the accuracy, prediction, recall values are calculated.
K=8,
![image](https://user-images.githubusercontent.com/39863708/122548501-10eea080-d04f-11eb-8d49-5fa6c1d5df98.png)
K=7,
![image](https://user-images.githubusercontent.com/39863708/122548519-164beb00-d04f-11eb-9229-009c8a64475f.png)
K=6, 
![image](https://user-images.githubusercontent.com/39863708/122548534-1c41cc00-d04f-11eb-97c2-8d3a3c925ce0.png)
Observation:-
It can be observed that the recall value is almost near by 82 percent which is great and that is why KNN is taken as final model.
9) Ensemble Learning:
![image](https://user-images.githubusercontent.com/39863708/122548568-282d8e00-d04f-11eb-8fe0-2a1adb6061c6.png)
![image](https://user-images.githubusercontent.com/39863708/122548573-2a8fe800-d04f-11eb-8bfd-126286304414.png)

In ensemble learning using voting and baggin classifier the precision in prediction value can be observed.
Voting classifier = 79.876 & Bagging classifier = 98.6867

here bagging classifier value is finally taken and with the test and training value it can be concluded.

Conclusion:-
so finally it can be concluded that the final precision value of the classifier is 98 percent which mean to predict stroke for percetn final result will give 98 percent accuracy.


