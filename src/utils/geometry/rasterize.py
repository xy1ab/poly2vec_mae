import numpy as np
import matplotlib.path as mpltPath


def rasterize_triangles(tris, spatial_size=256):
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T

    mask = np.zeros(spatial_size * spatial_size, dtype=bool)
    for tri in tris:
        path = mpltPath.Path(tri)
        mask |= path.contains_points(points)

    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)


def rasterize_polygon(coords, spatial_size=256):
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T

    path = mpltPath.Path(coords)
    mask = path.contains_points(points)
    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)
