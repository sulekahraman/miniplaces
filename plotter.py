import matplotlib.pyplot as plt
file=open("~/output.txt","r")
x=[]
y_val_1 = []
y_train_1 = []
y_val_5 = []
y_train_5 = []
i=0
firstline = True
for line in file: #range(10000):
    if len(line.split(" "))>2 and line.split(" ")[2] == "Training":
        train_line = line
        # print(i)
        print(train_line)
        train_accuracy_1 = float(train_line.split('Top1 = ')[-1].split(",")[0])
        train_accuracy_5 = float(train_line.split('Top5 = ')[-1].split(",")[0])
        Iter = int(train_line.split(" ")[1].strip(','))
        x.append(Iter)
        y_train_1.append(train_accuracy_1)
        y_train_5.append(train_accuracy_5)
 
    if len(line.split(" "))>2 and line.split(" ")[2] == "Validation":
        train_line = line
        # print(i)
        print(train_line)
        train_accuracy_1 = float(train_line.split('Top1 = ')[-1].split(',')[0])
        train_accuracy_5 = float(train_line.split('Top5 = ')[-1].split(',')[0])
        y_val_1.append(train_accuracy_1)
        y_val_5.append(train_accuracy_5)
 
 
plt.plot(x, y_train_1, label = "Top 1 Training Accuracy")  
plt.plot(x, y_val_1, label = "Top 1 Validation Accuracy")
plt.plot( x, y_train_5, label = "Top 5 Training Accuracy")
plt.plot(x, y_val_5, label  = "Top 5 Validation Accuracy")
plt.legend()
plt.show()