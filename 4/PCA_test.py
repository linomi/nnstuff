from PCA_extractor import PCA

x = np.array([[90,60,90],[90,90,30],[60,60,60],[60,60,90],[30,30,30]])
x = x - x.mean(0)
pc = PCA(x,3,1)
for i in range(0,100000):
    pc.update(alpha = 0.00000004)
print(pc.get_pcs())
