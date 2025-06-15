import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import griddata
from skimage.measure import marching_cubes
import pandas as pd
import os
import argparse

def get_distance_squared(p, x, y, z):
    return (p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2

def get_distance(p, x, y, z):
    return np.sqrt(get_distance_squared(p, x, y, z))

def plot_isosurface(points_with_values, level, spacing=6, grid_res=50, color='blue', name='isosurface'):
    x, y, z = points_with_values[:, 0], points_with_values[:, 1], points_with_values[:, 2]
    values = points_with_values[:, 3]

    xi = np.linspace(min(x), max(x), grid_res)
    yi = np.linspace(min(y), max(y), grid_res)
    zi = np.linspace(min(z), max(z), grid_res)
    grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')

    grid_values = griddata(points_with_values[:, :3], values, (grid_x, grid_y, grid_z), method='linear', fill_value=0)

    verts, faces, _, _ = marching_cubes(grid_values, level=level, spacing=(xi[1]-xi[0], yi[1]-yi[0], zi[1]-zi[0]))

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5, color=color, name=name
    )

def visualize_3d(points_with_values, Rs=None, vmin=None, vmax=None):
    x, y, z = points_with_values[:, 0], points_with_values[:, 1], points_with_values[:, 2]
    values = points_with_values[:, 3]

    traces = [
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=values, colorscale='Viridis', colorbar=dict(title='Signal'), opacity=0.8, cmin=vmin, cmax=vmax),
            name='Measurement Points')
    ]

    if Rs is not None:
        traces.append(go.Scatter3d(
            x=Rs[:, 0], y=Rs[:, 1], z=Rs[:, 2],
            mode='markers',
            marker=dict(size=6, color='red', symbol='x'),
            name='Wi-Fi Sources'))

    layout = go.Layout(title='3D Signal Visualization', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def generate_sample_data():
    base_distance = 6
    wifi_max = 100
    wifi_dist = 30
    R_num = np.random.randint(1, 8)
    Rs = np.random.rand(R_num, 3) * 40

    every_points = []
    def add_points(x_range, y_range, f_range, h_range):
        for f in f_range:
            for h in h_range:
                for x in x_range:
                    for y in y_range:
                        X = base_distance * x
                        Y = base_distance * y
                        Z = base_distance * (f * 4 + h)
                        closest = min(Rs, key=lambda p: get_distance_squared(p, X, Y, Z))
                        dist = get_distance(closest, X, Y, Z)
                        wifi = max(0, (wifi_dist - dist) / wifi_dist * wifi_max)
                        every_points.append([X, Y, Z, wifi])

    add_points(range(8, 10), range(10), range(2), range(3))
    add_points(range(2), range(10), range(2), range(3))
    add_points(range(2, 8), range(2), range(2), range(3))
    add_points(range(2, 8), range(5, 7), range(2), range(3))

    return np.array(every_points), Rs

def load_points_from_csv(filepath):
    df = pd.read_csv(filepath)
    if set(['x', 'y', 'z', 'metric']).issubset(df.columns):
        return df[['x', 'y', 'z', 'metric']].values
    else:
        raise ValueError("CSV file must have x, y, z, and metric columns")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--levels', type=int, nargs='+', default=[20, 60], help='List of signal boundaries')
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        every_points = load_points_from_csv(args.csv)
        Rs = None 
        print(f"Loaded {len(every_points)} points from CSV")
    else:
        every_points, Rs = generate_sample_data()
        print(f"Generated {len(every_points)} sample points")

    while True:
        surfaces = [
            plot_isosurface(every_points, level=lv, color='green' if lv >= 50 else 'red', name=f'Signal ≥ {lv}')
            for lv in args.levels
        ]

        fig = go.Figure(data=surfaces + [
            go.Scatter3d(x=every_points[:, 0], y=every_points[:, 1], z=every_points[:, 2],
                         mode='markers', marker=dict(size=2, color=every_points[:, 3], colorscale='Viridis', opacity=0.6),
                         name='Signal Points')
        ] + ([
            go.Scatter3d(x=Rs[:, 0], y=Rs[:, 1], z=Rs[:, 2], mode='markers', marker=dict(size=6, color='black', symbol='x'),
                         name='Wi-Fi Sources')
        ] if Rs is not None else []))

        fig.update_layout(title='Marching Cube Visualization', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

        try:
            user_input = input("Enter new reading (x y z value) or 'exit': ")
            if user_input.lower() == 'exit':
                break
            x, y, z, val = map(float, user_input.strip().split())
            distances = np.linalg.norm(every_points[:, :3] - np.array([x, y, z]), axis=1)
            nearest_idx = np.argmin(distances)
            every_points[nearest_idx, 3] = val
            print(f"Updated point at index {nearest_idx} to value {val}")
        except Exception as e:
            print(f"Invalid input: {e}")