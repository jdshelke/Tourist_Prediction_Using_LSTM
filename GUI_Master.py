# Import modules and packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from datetime import datetime
from subprocess import call
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

root = tk.Tk()
root.title("Tourist prediction")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('./images/taj.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Tourist Prediction System", font=('times', 35,' bold '), height=1, width=32,bg="violet Red",fg="Black")
lbl.place(x=300, y=10)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def add_records():
    from subprocess import call
    call(["python","add_records.py"])


def Model_Training():
    from subprocess import call
    call(["python","Model_training.py"])
    
    
def call_file():
    #import Check_Heart
    #Check_Heart.Train()
    country = tk.IntVar()
    month = tk.IntVar()
   
   
    def Detect():
        dataset_train = pd.read_csv('./Data/Tourist_Data_By_Months_2001-2020.csv')
        train_data = dataset_train['FTAs_in_India']
        # LSTM is sensitive to the scale of data so we apply min max scalar
        scaler = MinMaxScaler(feature_range = (0,1))
        scaled_data = scaler.fit_transform(np.array(train_data).reshape(-1,1))


        n_future = 12   # Number of days we want top predict into the future
        n_steps = 24     # Number of past days we want to use to predict the future

        # Convert an array of values into dataset marrix
        def create_dataset(Dataset, time_step=1):
            dataX, dataY = [],[]
            for i in range(len(Dataset)-time_step-1):
                a = Dataset[i:(i+time_step), 0]
                dataX.append(a)
                dataY.append(Dataset[i+time_step, 0])
            return np.array(dataX), np.array(dataY)

        # Reshaping into X=t, t+1, t+2, t+3 and Y=t+4
        time_step = 24
        X_train, y_train = create_dataset(scaled_data, time_step)

        # Reshape Input to be [Samples, time_step, Features] Which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        x_input=scaled_data[-24:].reshape(1,-1)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()


        e1=month.get()
        e1 = int(e1)


        #########################################################################################
        
        from tensorflow.keras.models import load_model
        model=load_model('./Models/tourist3.h5')
        # demonstrate prediction for next 12 Months
        from numpy import array

        lst_output=[]
        x1 = 490
        y1 = 420
        n_steps=24
        i=0
        while(i<e1):
            
            if(len(temp_input)>n_steps):
                y1 = y1+50
                if(i > 7):
                    x1 = 1000
                if(i == 8):
                    y1 = 420
                # x1 = x1+50
                #print(temp_input)
                x_input=np.array(temp_input[1:])
        #         print("{} Month input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                inverse_yhat = scaler.inverse_transform(yhat) 
                # print("{} Month output {}".format(i,inverse_yhat))
                
                # predicted_price=listToString(inverse_yhat[0]) 
                yes = tk.Label(root,text="Predicted Tourist in month "+ str(i+1) +str(inverse_yhat[0]),background="red",foreground="white",font=('times', 20, ' bold '))
                yes.place(x=x1,y=y1)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                inverse_yhat = scaler.inverse_transform(yhat) 
                # print(inverse_yhat[0])
                # predicted_price=listToString(inverse_yhat[0]) 
                yes = tk.Label(root,text="Predicted Tourist in month 1 "+str(inverse_yhat[0]),background="red",foreground="white",font=('times', 20, ' bold '))
                yes.place(x=x1,y=y1)
                temp_input.extend(yhat[0].tolist())
                # print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1

    l1=tk.Label(root,text="country",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=300,y=200)
    l3=tk.Label(root,text="India",background="#CCCCFF",font=('Times New Roman', 25, ' bold '),width=10)
    l3.place(x=500,y=200)

    l2=tk.Label(root,text="month",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=300,y=300)
    month=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=month)
    month.place(x=500,y=300)


    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=300,y=400)
    


def window():
    root.destroy()

# button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
# button2.place(x=5, y=90)

button5 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Add Records", command=add_records, width=15, height=2)
button5.place(x=5, y=170)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training, width=15, height=2)
button3.place(x=5, y=250)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Tourist_Prediction", command=call_file, width=15, height=2)
button4.place(x=5, y=330)

exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=410)

root.mainloop()



