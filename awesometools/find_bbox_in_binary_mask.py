'''
在二值mask（只含有单个目标）中找到目标的bbox
'''
import time
import numpy as np

# Setup
mask = np.zeros((4000, 5000), dtype=np.bool8)
for j in range(-50, 50):
    for k in range(-50, 50):
        if np.sqrt(j**2 + k**2) < 50:
            mask[1000 + j, 2000 + k] = True
image = np.random.randint(0, 256, (4000, 5000, 3), dtype=np.uint8)
exp_image = image[951:1050, 1951:2050]
n_loops = 1000

# Method 1
t1 = time.time()
for _ in range(n_loops):
    i, j = np.where(mask)
    y, x = np.meshgrid(
        np.arange(min(i), max(i) + 1),
        np.arange(min(j), max(j) + 1),
        indexing="ij",
    )
    sub_image = image[y, x]
t2 = time.time()
print((t2 - t1) / n_loops)
assert sub_image.shape == exp_image.shape, (sub_image.shape, exp_image.shape)
assert np.all(sub_image == exp_image)

# Method 2
t1 = time.time()
for _ in range(n_loops):
    where = np.array(np.where(mask))
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1) + 1
    sub_image = image[y1:y2, x1:x2]
t2 = time.time()
print((t2 - t1) / n_loops)
print(x1, y1, x2, y2)
assert sub_image.shape == exp_image.shape, (sub_image.shape, exp_image.shape)
assert np.all(sub_image == exp_image)

# Method 3
t1 = time.time()
for _ in range(n_loops):
    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
    sub_image = image[y1:y2, x1:x2]
t2 = time.time()
print((t2 - t1) / n_loops)
print(x1, y1, x2, y2)
assert sub_image.shape == exp_image.shape, (sub_image.shape, exp_image.shape)
assert np.all(sub_image == exp_image)