from tkinter import *
import tkinter as tk
from tkinter import messagebox as mb
from tkcalendar import DateEntry
from PIL import Image, ImageTk
import datetime
import csv

root = Tk()
root.geometry('520x540')
root.title("ADD RECORDS")
root.configure(background='grey')

image2 = Image.open('./images/add_records.jpg')
image2 = image2.resize((520, 540), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)

#defining function msg() using messagebox
def msg():
    # get the index of the last character in the widget,if it is zero,it is empty
    # if (e1.index("end") == 0):
    mb.showwarning('Missing details', 'Enter The Tourist Count')
    # else:
    #     mb.showinfo('Success', 'Records are added Successfully! ')


#exporting entered data
def save():
    # Taking The Previous Date from Dataset
    from dateutil.relativedelta import relativedelta
    with open("./Data/Tourist_Data_By_Months.csv", "r") as f:
        last_line = f.readlines()[-1]
        first = last_line.split(",")[0]
        latest = datetime.datetime.strptime(first, "%Y/%m/%d")
        date_after_month = latest + relativedelta(months=1)
        f.close()
    
    #save data in csv file

    with open('./Data/Tourist_Data_By_Months.csv', 'a', newline='') as fs:
        w = csv.writer(fs)
        w.writerow([date_after_month.strftime("%Y/%m/%d"), e1.get()])
        fs.close()
    
    l2 = Label(root, width=20,text = date_after_month.strftime("%Y/%m/%d"), bg='#CCCCFF', fg='black', font=("Times New Roman",15,"bold"))
    l2.place(x=240,y=230)

def saveinfo():
    try:
        int(e1.get())
        save()
        delete(e1)
    except ValueError:
        msg()

# creating labels and entry widgets

def delete(label):
    label.delete(0, 'end')

l1 = Label(root, text="ADD RECORDS",font=("Times New Roman", 25, 'bold'),background="#CCCCFF", fg="black", width=20)
l1.place(x=70,y=50)


l4 = Label(root, text="Date",width=10,font=("Times New Roman",15,"bold"),anchor="w",bg='#CCCCFF', fg="black")
l4.place(x=70,y=230)


l6 = Label(root, text="Tourist",width=10,font=("Times New Roman",15,"bold"),anchor="w",bg='#CCCCFF')
l6.place(x=70,y=320)
e1 = Entry(root,width=30,bd=5)
e1.place(x=240,y=320)


# submit and cancel buttons
b1 = Button(root, text='Submit',command=saveinfo,width=15,bg='green',fg='white',font=("times",12,"bold"))
b1.place(x=120,y=440)
b2 = Button(root, text='Cancel',command=root.destroy,width=15,bg='maroon',fg='white',font=("times",12,"bold"))
b2.place(x=320,y=440)

root.mainloop()