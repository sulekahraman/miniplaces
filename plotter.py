import matplotlib.pyplot as plt
# file=open("~/output.txt","r")
x=list(range(1,11))


top5ogTraining = [9.445,16.216,20.276,23.073,25.565,27.47,29.055,30.316,31.562,32.535]
top5ogValidation = [13.97,18.91,22.24,25.17,27.33,28.33,30.33,31.72,32.72,33.58]


top5modifiedTraining = [34.093,52.834,62.039,67.7,71.627,74.795,77.889,80.611,83.407,86.064]
top5modifiedValidation = [44.19,55.33,60.5,62.39,68.33,68.13,66.44,67.15,68.14,64.39]



# top5modifiedTraining = [8.283,11.638,14.413,17.503,20.629,23.559,25.982,28.084,29.969,31.731]
# top5modifiedValidation = [10.99,12.75,16.4,19.38,22.3,24.6,26.8,29.48,29.58,33.57]


# top5modifiedTraining = [5.627,5.61,5.605,5.654,5.625,5.618,5.607,5.655,5.64,5.679]
# top5modifiedValidation = [5.64,5.66,5.62,5.57,5.57,5.58,5.61,5.57,5.51,5.55]

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

top5modifiedTraining = subtract100(top5modifiedTraining)
top5modifiedValidation = subtract100(top5modifiedValidation)


plt.plot(x, top5ogTraining, label = "Baseline Top-5 Error on Training Set")  
plt.plot(x, top5ogValidation, label = "Baseline Top-5 Error on Validation Set")
plt.plot(x, top5modifiedTraining, label = "Trial Run Top-5 Error on Training Set")  
plt.plot(x, top5modifiedValidation, label = "Trial Run Top-5 Error on Validation Set")
plt.legend()
plt.show()