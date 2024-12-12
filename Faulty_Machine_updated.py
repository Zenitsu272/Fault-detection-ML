
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib #to save our model 

from sklearn.ensemble import RandomForestClassifier
#Data cleaning


data=pd.read_csv("data1.csv")


#checking null values in the dataset 

#print(data.isnull().any()) #no null values in the data set.

#checking for duplicate entries 
#print(data.duplicated())

#dropping the duplicates if any.
data=data.drop_duplicates()


'''building the model'''

x=data.drop(columns=["fail"])#this drops the fail column alone and has all other column with it 

y=data["fail"] #now the fail column alone is being present in this 


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3,random_state=42)

#training the model

model = RandomForestClassifier(n_estimators=100,max_depth=5, random_state=42)

model.fit(xtrain,ytrain)



'''   #to understand ytrain,test,xtrain,test run the following 4 lines 
print(xtrain)
print(xtest)
print("y train \n",ytrain)
print("y test ",ytest)
'''
prediction=model.predict(xtest)

for i in range(len(prediction)):
    if prediction[i]==1:
        print("high probability for fault for sample ",i+1)
    elif prediction[i]==0:
        print("probability of fault is low for sample ",i+1)

joblib.dump(model,"Faulty_Machinerandom.joblib")
#calculating the accuracy of the model
prediction=model.predict(xtest)
accuracy=accuracy_score(ytest,prediction)
print("testing accuracy of the model is :",accuracy*100,"percentage")

training_prediction=model.predict(xtrain)
tr_accuracy=accuracy_score(ytrain,training_prediction)
print("training accuracy :",tr_accuracy*100)



print("getting new predictions from the user :")
lis=[]
n=int(input("enter number of inputs :"))
for i in range(n):
    ff=int(input("enter footfall: "))
    tm=int(input("enter tempmode:"))
    aq=int(input("enter aq:"))
    uss=int(input("enter uss:"))
    cs=int(input("enter cs :"))
    voc=int(input("enter voc :"))
    rp=int(input("enter rp:"))
    ip=int(input("enter ip:"))
    temp=int(input("enter temperature :"))
    
    
    lis.append([ff, tm, aq, uss, cs, voc, rp, ip,temp])
    
new_predictions=model.predict(lis)
print("new predictions :",new_predictions)


for i in range(n):
    if (new_predictions[i]==0):
        print("probability of fault is less ")
    elif (new_predictions[i]==1):
        print("probability of fault is high ")
        
''''joblib.dump(model,"Faulty_Machine.joblib") #to save the model'''

'''if you wanna load this model next time, 
import joblib
model = joblib.load("Faulty_Machine.joblib")

tree.export_graphviz(
    model,
    out_file="Faulty_Machine1.dot",
    feature_names=["footfall","tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"],
   # class_names=sorted(y.unique()),
    label="all",
    rounded=True,
    filled=True,
)
'''

'''import graphviz

# Read and render the .dot file into an image format (e.g., PNG)
with open("Faulty_Machine1.dot") as f:
    dot_source = f.read()

graph = graphviz.Source(dot_source)
graph.render("Faulty_Machine1", format="png", view=True)
'''


























