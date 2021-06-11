import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from datetime import datetime
from subprocess import call
import tkinter as tk
from tkinter import messagebox as mb
from PIL import Image, ImageTk
# from tkinter import ttk
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder


root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Tourist_Prediction")

#####For background Image######
image2 = Image.open('./images/taj.jpg')
image2 = image2.resize((1530, 900), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)

label_l1 = tk.Label(root, text="Model Training And Visualization",font=("Times New Roman", 35, 'bold'),
                    background="#CCCCFF", fg="black", width=30, height=1)
label_l1.place(x=300, y=10)

model_name = tk.StringVar()
Epoch = tk.IntVar()
future_months = tk.IntVar()


def data_Preprocessing():
    global datelist_train, train_data, X_train, y_train, scaler, x_input, scaled_data
    # Importing Training Set
    dataset_train = pd.read_csv('./Data/Tourist_Data_By_Months_2001-2020.csv')
    # Select features (columns) to be involved intro training and predictions
    cols = list(dataset_train)[1]

    # Extract dates (will be used in visualization)
    datelist_train = list(dataset_train['Date'])
    datelist_train = [dt.datetime.strptime(date, '%Y/%m/%d').date() for date in datelist_train]

    # print(dataset_train.shape)
    train_data = dataset_train['FTAs_in_India']

    # LSTM is sensitive to the scale of data so we apply min max scalar
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(np.array(train_data).reshape(-1,1))
    # print(scaled_data.shape)

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
    # print(X_train.shape)

    
 
def train():
    global line2
    e1=Epoch.get()
    e1 = int(e1)
    # Model_Name = model_name.get()
    path_for_model = './Models/'
    model_path = os.path.join(path_for_model, model_name.get()) + '.h5'
    # print(model_path)
    data_Preprocessing()
    # Import Libraries and packages from Keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.optimizers import Adam

    # Initializing the Neural Network based on LSTM
    model = Sequential()
    # Adding 1st LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(24, 1)))
    # Adding 2nd LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    # Adding 2nd LSTM layer
    model.add(LSTM(50))
    # Adding Dropout
    model.add(Dropout(0.25))
    # Output layer
    model.add(Dense(units=1, activation='linear'))
    # Compiling the Neural Network
    model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

    # Train The Model
    model.fit(X_train, y_train, epochs=e1, verbose=1)

    model.save(model_path)
    
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    
    total_predict = model.predict(X_train)
    # Transform Back To Original Form
    total_predict = scaler.inverse_transform(total_predict) 

    # Plot Graph
    figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, root)
    # line2.get_tk_widget().pack(side=tk.LEFT)
    line2.get_tk_widget().place(x=490,y=220)
    ax2.plot(train_data,'red')
    ax2.plot(total_predict,'blue')
    ax2.set_ylabel('Tourist Count')
    ax2.set_title('Model Training')

def future_prediction():
    global line1
    path_for_model = './Models/'
    model_path = os.path.join(path_for_model, model_name.get()) + '.h5'
    try:
        model = load_model(model_path)
    except OSError:
        mb.showwarning('Missing details', 'Enter The Valid Model Name')
        
    data_Preprocessing()
    n_future = future_months.get()
    n_steps = 24

    x_input=scaled_data[-24:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 12 Months
    from numpy import array

    lst_output=[]
    i=0
    while(i<n_future):
        if(len(temp_input)>n_steps):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            inverse_yhat = scaler.inverse_transform(yhat) 
            # print("{} Month output {}".format(i,inverse_yhat))
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
            temp_input.extend(yhat[0].tolist())
    #         print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    
    # Generate list of sequence of days for predictions
    datelist_future = pd.date_range(datelist_train[-1] , periods=future_months.get(), freq='m').tolist()

    # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())
    
    month_new=datelist_train[-24:]
    month_pred=datelist_future_

    # rcParams['figure.figsize'] = 14, 5
    figure1 = plt.Figure(figsize=(10,5), dpi=100)
    ax1 = figure1.add_subplot(111)
    line1 = FigureCanvasTkAgg(figure1, root)
    # line2.get_tk_widget().pack(side=tk.LEFT)
    line1.get_tk_widget().place(x=290,y=220)
    ax1.plot(month_new,scaler.inverse_transform(scaled_data[204:]))
    ax1.plot(month_pred,scaler.inverse_transform(lst_output))
    ax1.legend(shadow=True)
    # plt.title('Predcitions and Acutal Tourist', family='Arial', fontsize=12)
    ax1.set_ylabel('Tourist Count')
    ax1.set_title('Future Predcitions and Acutal Tourist')
    # plt.xlabel('Timeline', family='Arial', fontsize=10)
    # plt.ylabel('Tourist Count', family='Arial', fontsize=10)
    # ax1.xticks(rotation=45, fontsize=8)
    # ax1.ticklabel_format(style='plain', axis = 'y')

def delete(text):
    text.delete(0, 'end')

def destroy(label):
    label.destroy()

def run_train():
    try:
        if(model_name.get() == '' or Epoch.get() == 0):
            raise Exception("Sorry, numbers zero is not allow and Empty string not allowed")
        int(Epoch.get())
        train()
        delete(model_name)
    except ValueError:
        mb.showwarning('Missing details', 'Enter The Number of Iterations')
    except Exception:
        mb.showwarning('Missing details', 'Enter The Model Name And Epoch Greater than 0')


def run_future_prediction():
    try:
        if(model_name.get() == '' or future_months.get() == 0):
            raise Exception("Sorry, Empty string not allowed")
        int(future_months.get())
        future_prediction()
        delete(model_name)
        delete(month)
    except ValueError:
        mb.showwarning('Missing details', 'Enter The Number of Next Months')
    except Exception:
        mb.showwarning('Missing details', 'Enter The Model Name And Next Months Greater than 0')


def window():
    root.destroy()


l1=tk.Label(root,text="Model Name",foreground='white',background="purple",font=('times', 20, ' bold '),width=10)
l1.place(x=5,y=130)
model_name=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=model_name)
model_name.place(x=200,y=130)
l2=tk.Label(root,text="Epoch",foreground='white',background="purple",font=('times', 20, ' bold '),width=10)
l2.place(x=5,y=180)
Epoch=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Epoch)
Epoch.place(x=200,y=180)
button5 = tk.Button(root, foreground="black", background="blue", font=("Tempus Sans ITC", 14, "bold"),
                    text="Train Model", command=run_train, width=15, height=2)
button5.place(x=5, y=230)
l3=tk.Label(root,text="Next Months",foreground='white',background="purple",font=('times', 20, ' bold '),width=10)
l3.place(x=5,y=360)
month=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=future_months)
month.place(x=200,y=360)
button5 = tk.Button(root, foreground="black", background="blue", font=("Tempus Sans ITC", 14, "bold"),
                    text="Future Prediction", command=run_future_prediction, width=15, height=2)
button5.place(x=5, y=410)

exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=490)

root.mainloop()