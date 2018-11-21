import matplotlib.pyplot as plt
# file=open("~/output.txt","r")
x=list(range(1,11))


top5ogTraining = [9.445,16.216,20.276,23.073,25.565,27.47,29.055,30.316,31.562,32.535]
top5ogValidation = [13.97,18.91,22.24,25.17,27.33,28.33,30.33,31.72,32.72,33.58]

top5modifiedTraining = []
top5modifiedValidation = []

firstline = True
# for line in file: #range(10000):
#     if len(line.split(" "))>2 and line.split(" ")[2] == "Training":
#         train_line = line
#         # print(i)
#         print(train_line)
#         train_accuracy_1 = float(train_line.split('Top1 = ')[-1].split(",")[0])
#         train_accuracy_5 = float(train_line.split('Top5 = ')[-1].split(",")[0])
#         Iter = int(train_line.split(" ")[1].strip(','))
#         x.append(Iter)
#         y_train_1.append(train_accuracy_1)
#         y_train_5.append(train_accuracy_5)
 
#     if len(line.split(" "))>2 and line.split(" ")[2] == "Validation":
#         train_line = line
#         # print(i)
#         print(train_line)
#         train_accuracy_1 = float(train_line.split('Top1 = ')[-1].split(',')[0])
#         train_accuracy_5 = float(train_line.split('Top5 = ')[-1].split(',')[0])
#         y_val_1.append(train_accuracy_1)
#         y_val_5.append(train_accuracy_5)


def subtract100(vals):
    output = []
    for val in vals:
        output.append( 100 - val )
    return output


top5ogTraining = subtract100(top5ogTraining)
top5ogValidation = subtract100(top5ogValidation)


subtract100(top5ogTraining)
plt.plot(x, top5ogTraining, label = "Baseline Top-5 Error on Training Set")  
plt.plot(x, top5ogValidation, label = "Baseline Top-5 Error on Validation Set")
# plt.plot( x, y_train_5, label = "Top 5 Training Accuracy")
# plt.plot(x, y_val_5, label  = "Top 5 Validation Accuracy")
plt.legend()
plt.show()