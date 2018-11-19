import matplotlib.pyplot as plt
import json

#Load error dictionaries
with open("output/train_top1.json","r") as t1:
    train1 = json.load(t1)
with open("output/train_top5.json","r") as t5:
    train5 = json.load(t5)

with open("output/val_top1.json","r") as v1:
    val1 = json.load(v1)
with open("output/val_top5.json","r") as v5:
    val5 = json.load(v5)

train_top1 = [value for (key, value) in sorted(train1.items())]
train_top5 = [value for (key, value) in sorted(train5.items())]

val_top1 = [value for (key, value) in sorted(val1.items())]
val_top5 = [value for (key, value) in sorted(val5.items())]

x = arange(1, 11)
plt.plot(x, train_top1, label = "Top-1 Training Error")  
plt.plot(x, val_top1, label = "Top-1 Validation Error")
plt.plot(x, train_top5, label = "Top-5 Training Error")
plt.plot(x, val_top5, label  = "Top-5 Validation Error")
plt.legend()
plt.show()