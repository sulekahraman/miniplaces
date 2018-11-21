import matplotlib.pyplot as plt
import json

#Load error dictionaries
# with open("output/default/train_top1.json","r") as t1:
#     train1 = json.load(t1)
# with open("output/val_top1.json","r") as v1:
#     val1 = json.load(v1)
with open("output/default/train_top5.json","r") as t5:
    train5 = json.load(t5)
with open("output/default/val_top5.json","r") as v5:
    val5 = json.load(v5)
with open("output/exp5/val_top5.json","r") as v:
    mod_train5 = json.load(v)
with open("output/exp5/train_top5.json","r") as t:
    mod_val5 = json.load(t)

# train_top1 = [value for (key, value) in sorted(train1.items())]
train_top5 = [value for (key, value) in sorted(train5.items(), key=lambda x: int(x[0]))]
mod_train_top5 = [value for (key, value) in sorted(mod_train5.items(), key=lambda x: int(x[0]))]

# val_top1 = [value for (key, value) in sorted(val1.items())]
val_top5 = [value for (key, value) in sorted(val5.items(), key=lambda x: int(x[0]))]
mod_val_top5 = [value for (key, value) in sorted(mod_val5.items(), key=lambda x: int(x[0]))]

print(train_top5)
x = range(1, len(train_top5)+1)
# plt.plot(x, train_top1, label = "Top-1 Training Error")  
# plt.plot(x, val_top1, label = "Top-1 Validation Error")
plt.plot(x, train_top5, label = "Training Err of Original")
plt.plot(x, val_top5, label  = "Validation Err of Original")
plt.plot(x, mod_train_top5[:10], label = "Training Err of Modified")
plt.plot(x, mod_val_top5[:10], label  = "Validation Err of Modified")
plt.xlabel("epoch")
plt.ylabel("Top-5 Error (%)")
plt.title("SGD Optimizer original:lr=1e-3, modified lr=0.1, with scheduler, and weight_decay=5e-4")
plt.legend()
plt.show()