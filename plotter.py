import matplotlib.pyplot as plt
import json

#Load error dictionaries
# with open("output/default/train_top1.json","r") as t1:
#     train1 = json.load(t1)
with open("output/lr_01/adam/train_top5.json","r") as t5:
    train5 = json.load(t5)

# with open("output/val_top1.json","r") as v1:
#     val1 = json.load(v1)
with open("output/lr_01/adam/val_top5.json","r") as v5:
    val5 = json.load(v5)

# train_top1 = [value for (key, value) in sorted(train1.items())]
train_top5 = [value for (key, value) in sorted(train5.items())]

# val_top1 = [value for (key, value) in sorted(val1.items())]
val_top5 = [value for (key, value) in sorted(val5.items())]

x = range(1, len(train_top5)+1)
# plt.plot(x, train_top1, label = "Top-1 Training Error")  
# plt.plot(x, val_top1, label = "Top-1 Validation Error")
plt.plot(x, train_top5, label = "Top-5 Training Error")
plt.plot(x, val_top5, label  = "Top-5 Validation Error")
plt.xlabel("epoch")
plt.ylabel("Top-5 Error")
plt.title("Adam Optimizer with lr=0.1, weight_decay=5e-4")
plt.legend()
plt.show()