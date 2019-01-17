import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from hb_report import hb_normal

x, y = np.meshgrid(np.linspace(-200, 100, 10), np.linspace(-200, 100, 10))
zs = np.array([hb_normal([x, y])[1] for x, y in zip(np.ravel(x), np.ravel(y))])
mask = np.array([x > y for x, y in zip(np.ravel(x), np.ravel(y))])
z = zs.reshape(x.shape)
z = np.ma.array(z, mask=mask)
fig, ax = plt.subplots()
cs = ax.contourf(x, y, z, corner_mask=True)
ax.contour(cs, colors='k')
cbar = plt.colorbar(cs)
# im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(-3, 3, -2, 2))
# cbar = plt.colorbar(im)
plt.show()
