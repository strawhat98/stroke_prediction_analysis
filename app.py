import sklearn 
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(age, Marital_Status,hypertension, heart_disease, avg_glucose_level, height, weight,smoking_status,work_type):   
 
    # Pre-processing user input    
 
    if Marital_Status == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if hypertension == "No":
        ht = 0
    else:
        ht = 1  
        
    if heart_disease == "No":
        hd = 0
    else:
        hd = 1 
        
    avg_glucose_level = avg_glucose_level / 100
    bmi = (weight/(height * height))/10
    if smoking_status == "Formerly smoked":
        sst = 1
    elif smoking_status == "never smoked":
        sst = 2
    elif smoking_status == "smokes":
        sst = 3
    else:
        sst=0
    if work_type == "Private":
        wt = 2
    elif work_type == "Self-employed":
        wt = 3
    elif work_type == "Govt_job":
        wt = 0
    elif work_type == "children":
        wt = 4
    else: 
        wt = 1


    # Making predictions 
    prediction = classifier.predict( 
        [[age, Married,ht, hd, avg_glucose_level, bmi,sst,wt]])
     
    #if prediction == 0:
    #    pred = 'Rejected'
    #else:
    #    pred = 'Approved'
    return prediction
    
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Stroke Prediction APP</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Age = st.number_input("age(3-99)")
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    Hypertension = st.selectbox('hypertension',("Yes","No"))
    HeartDisease = st.selectbox('heart_disease',("Yes","No"))
    AGL = st.number_input("average_glucose_level") 
    H = st.number_input("height in meters")
    W = st.number_input("weight in kg")
    Smoke = st.selectbox('Smoking Status',("Formerly smoked","never smoked","smokes","Unknown"))
    Work = st.selectbox('Work Status',("Private","Self-Employed","Govt_job","Children","Never_worked"))
    result = 0
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Age, Married, Hypertension, HeartDisease, AGL, H, W,Smoke,Work) 
        st.success(print(result))
        #print(predict)
     
if __name__=='__main__': 
    main()

