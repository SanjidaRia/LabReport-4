import random

class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.group = None

class ManhattanKMeans:
    def __init__(self, total_points=100, total_clusters=10, plane_size=30):
        self.total_points = total_points
        self.total_clusters = total_clusters
        self.plane_size = plane_size
        self.data = []
        self.centers = []
        self.generate_data()
        self.initialize_clusters()
        self.cluster_data()

    def generate_data(self):
        self.data = [DataPoint(random.randint(0, self.plane_size - 1),
                               random.randint(0, self.plane_size - 1)) for _ in range(self.total_points)]

    def initialize_clusters(self):
        self.centers = [DataPoint(random.randint(0, self.plane_size - 1),
                                  random.randint(0, self.plane_size - 1)) for _ in range(self.total_clusters)]

    def compute_manhattan(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)

    def assign_to_clusters(self):
        for point in self.data:
            distances = [self.compute_manhattan(point, center) for center in self.centers]
            point.group = distances.index(min(distances))

    def recalculate_centers(self):
        for idx in range(self.total_clusters):
            members = [p for p in self.data if p.group == idx]
            if members:
                avg_x = round(sum(p.x for p in members) / len(members))
                avg_y = round(sum(p.y for p in members) / len(members))
                self.centers[idx].x = avg_x
                self.centers[idx].y = avg_y

    def cluster_data(self):
        epoch = 0
        while True:
            previous = [(center.x, center.y) for center in self.centers]
            self.assign_to_clusters()
            self.recalculate_centers()
            current = [(center.x, center.y) for center in self.centers]
            epoch += 1
            if previous == current or epoch >= 100:
                break
        self.display_results()

    def display_results(self):
        grid = [["." for _ in range(self.plane_size)] for _ in range(self.plane_size)]

        for center_idx, center in enumerate(self.centers):
            grid[center.y][center.x] = chr(65 + center_idx)  # Cluster centers labeled A, B, C...

        for point in self.data:
            if grid[point.y][point.x] == ".":
                grid[point.y][point.x] = str(point.group)

        print("\n--- Grid View of Clusters ---")
        for line in grid:
            print(" ".join(line))

        print("\n--- Cluster Centers (A, B, C, ...) ---")
        for idx, center in enumerate(self.centers):
            print(f"Cluster {idx} (Center {chr(65 + idx)}): ({center.x}, {center.y})")

        print("\n--- Data Points and Assigned Clusters ---")
        for idx, point in enumerate(self.data):
            print(f"Point {idx}: ({point.x}, {point.y}) -> Cluster {point.group}")

if __name__ == "__main__":
    clustering = ManhattanKMeans()
