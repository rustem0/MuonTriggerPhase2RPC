# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from sklearn.neighbors import KernelDensity

# create data
x = np.random.normal(size=500)
y = x * 3 + np.random.normal(size=500)
 
# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins=300
k = kde.gaussian_kde([x,y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

print(xi)
print(yi)

zi = k(np.vstack([xi.flatten(), yi.flatten()]))

kde = KernelDensity(kernel='epanechnikov', bandwidth=0.2)
fit = kde.fit([x, y])
scores = fit.score(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, scores.reshape(xi.shape), shading='auto')
plt.show()
 
