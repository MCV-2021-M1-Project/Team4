import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def recall(p,f1):
    r = p*f1/(2*p-f1)
    return r

def all_measures(r,r2,p,p2):
    all_p = np.append(p.flatten(),p2.flatten())
    all_r = np.append(r.flatten(),r2.flatten())

    p_sorted = np.sort(all_p.flatten())
    index = np.argwhere(p_sorted == 0)

    r_sorted = all_r.flatten()[np.argsort(all_p.flatten())]
    indexes = np.argwhere(p_sorted < 0.9)
    p_sorted = np.delete(p_sorted,indexes)
    r_sorted = np.delete(r_sorted,indexes)
    
    
    """ p_sorted = np.delete(p_sorted,index)
    r_sorted = np.delete(r_sorted,index) """
    
    return p_sorted, r_sorted
    
"""Saturation + Value"""
with open("precision.npy","rb") as f:
    precisions = np.load(f)
    
with open("recall.npy","rb") as f:
    recalls = np.load(f)
    
with open("f1_measures.npy","rb") as f:
    f1_measures = np.load(f)
    
with open("precision2.npy","rb") as f:
    precisions2 = np.load(f)
    
with open("recall2.npy","rb") as f:
    recalls2 = np.load(f)
    
with open("f1_measures2.npy","rb") as f:
    f1_measures2 = np.load(f)
    
"""Saturation"""
with open("precision_S.npy","rb") as f:
    precisions_S = np.load(f)
    
with open("recall_S.npy","rb") as f:
    recalls_S = np.load(f)
    
with open("f1_measures_S.npy","rb") as f:
    f1_measures_S = np.load(f)
    
with open("precision_S2.npy","rb") as f:
    precisions_S2 = np.load(f)
    
with open("recall_S2.npy","rb") as f:
    recalls_S2 = np.load(f)
    
with open("f1_measures_S2.npy","rb") as f:
    f1_measures_S2 = np.load(f)
    
    
"""Value"""
with open("precision_V.npy","rb") as f:
    precisions_V = np.load(f)
    
with open("recall_V.npy","rb") as f:
    recalls_V = np.load(f)
    
with open("f1_measures_V.npy","rb") as f:
    f1_measures_V = np.load(f)
    
"""YCrCb - Y"""
with open("precision_y.npy","rb") as f:
    precisions_y = np.load(f)
    
with open("recall_y.npy","rb") as f:
    recalls_y = np.load(f)
    
with open("f1_measures_y.npy","rb") as f:
    f1_measures_y = np.load(f)
    
with open("precision_y2.npy","rb") as f:
    precisions_y2 = np.load(f)
    
with open("recall_y2.npy","rb") as f:
    recalls_y2 = np.load(f)
    
with open("f1_measures_y2.npy","rb") as f:
    f1_measures_y2 = np.load(f)
    
    
"""CieLab - L"""
with open("precision_L.npy","rb") as f:
    precisions_L = np.load(f)
    
with open("recall_L.npy","rb") as f:
    recalls_L = np.load(f)
    
with open("f1_measures_L.npy","rb") as f:
    f1_measures_L = np.load(f)
    
"""Grayscale"""
with open("precision_gray.npy","rb") as f:
    precisions_gray = np.load(f)
    
with open("recall_gray.npy","rb") as f:
    recalls_gray = np.load(f)
    
with open("f1_measures_gray.npy","rb") as f:
    f1_measures_gray = np.load(f)
    
""" print(np.sort(f1_measures2)) """

""" print("\nValue: 61, Saturation: 114")
print(f1_measures2[33,14])
print(precisions2[33,14])
print(recalls2[33,14])

print("\nValue: 62, Saturation: 114")
print(f1_measures2[32,14])
print(precisions2[32,14])
print(recalls2[33,14])

print("\nValue: 61, Saturation: 114")
print(f1_measures2[31,14])
print(precisions2[31,14])
print(recalls2[31,14]) """
    
    
""" fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
img1 = plt.imshow(np.flip(precisions,axis=0),extent=[0,250,0,250])
cbar1 = plt.colorbar(img1,ticks=[0.5,0.6, 0.7, 0.8, 0.9], orientation='horizontal')
cbar1.ax.set_xticklabels(['50%','60%', '70%', '80%','90%'])
ax1.set_ylabel("Value")
ax1.set_xlabel("Saturation")
ax1.set_title("Precision")
ax2 = fig.add_subplot(1,3,2)
img2 =plt.imshow(np.flip(recalls,axis=0),extent=[0,250,0,250])
cbar2 = plt.colorbar(img2,ticks=[0.5,0.6, 0.7, 0.8, 0.9], orientation='horizontal')
cbar2.ax.set_xticklabels(['50%','60%', '70%', '80%','90%'])
ax2.set_ylabel("Value")
ax2.set_xlabel("Saturation")
ax2.set_title("Recall")

ax3 = fig.add_subplot(1,3,3)
img3 = plt.imshow(np.flip(f1_measures,axis=0),extent=[0,250,0,250])
cbar3 = fig.colorbar(img3,ticks=[0.5,0.6, 0.7, 0.8, 0.9], orientation='horizontal')
cbar3.ax.set_xticklabels(['50%','60%', '70%', '80%','90%'])
ax3.set_ylabel("Value")
ax3.set_xlabel("Saturation")
ax3.set_title("F1 Measure") """


plt.figure()



#PR curves
p_sorted, r_sorted = all_measures(recalls,recalls2,precisions,precisions2)
plt.plot(p_sorted, r_sorted, label = 'Saturation + Value')

ps_sorted, rs_sorted = all_measures(recalls_S,recalls_S2,precisions_S,precisions_S2)
plt.plot(ps_sorted,rs_sorted, label = "Saturation",linewidth=2)

""" pv_sorted,rv_sorted = all_measures(recalls_V,np.array([]),precisions_V,np.array([]))
plt.plot(pv_sorted,rv_sorted, label = "Value") """

py_sorted, ry_sorted = all_measures(recalls_y,recalls_y2,precisions_y,precisions_y2)
plt.plot(py_sorted,ry_sorted, label = "Y - YCrCb")

pl_sorted, rl_sorted = all_measures(recalls_L,np.array([]),precisions_L,np.array([]))
plt.plot(py_sorted,ry_sorted, label = "L - CieLab")

pgray_sorted, rgray_sorted = all_measures(recalls_gray,np.array([]),precisions_gray,np.array([]))
plt.plot(py_sorted,ry_sorted, label = "Grayscale",linewidth=2)

plt.ylabel("Recall")
plt.xlabel("Precision")


all_recalls = []
all_precisions = []

#Draw F1-measures lines
for f1 in range(70,100,7):
    recall_f1 = []
    precision_f1 = []
    for p in range(900,1000,10):
        if p/500>f1/100:
            r = recall(p/1000,f1/100)
            if r <= 1 and r >= 0:
                precision_f1.append(p/1000)
                recall_f1.append(r)
    
    plt.text(precision_f1[-1],recall_f1[-1],"f1 = " + str(f1/100))
    plt.plot(precision_f1,recall_f1, 'b:',linewidth=0.5)
    
plt.legend()
plt.show()
