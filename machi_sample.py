import numpy as np
import pandas as pd

def get_distance_squared(p, x, y, z):
    return (p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2

def get_distance(p, x, y, z):
    return np.sqrt(get_distance_squared(p, x, y, z))

def generate_sample_data():
    base_distance = 6
    wifi_max = 100
    wifi_dist = 30
    R_num = 4 #np.random.randint(1, 8)
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

    return np.array(every_points)

if __name__ == '__main__':
    points = generate_sample_data()
    df = pd.DataFrame(points, columns=['x', 'y', 'z', 'metric'])
    df.to_csv('sample_wifi_data.csv', index=False)
    print(f"Saved {len(df)} rows to 'sample_wifi_data.csv'")