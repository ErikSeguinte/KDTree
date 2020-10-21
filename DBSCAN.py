from scipy.spatial import cKDTree


class DBSCAN:
    def __init__(self, min_pts = 4, distance=0.1, protocol=0):
        self.q = set()
        self.memo = {}
        self.visited = set()
        self.min_pts = min_pts
        self.distance = distance
        self.clusters = []
        self.points = []
        self.tree = None
        self.clusters_used = set()

    def range_query(self, i, eps, min, cluster):

        if i in self.visited:
            return
        self.visited.add(i)

        point = self.points[i]
        neighbors = self.tree.query_ball_point(x=point, r=eps, n_jobs=-1)

        if len(neighbors) < min and cluster not in self.clusters_used:
            # else:
            self.clusters[i] = 0
        else:
            self.clusters[i] = cluster
            self.clusters_used.add(cluster)

            if len(neighbors) >= min:
                for p in neighbors:
                    if p not in self.q:
                        self.q.add(p)
                        self.range_query(p, eps, min, cluster)

    def dbscan(self, eps, min):
        self.clusters = [0] * len(self.points)
        self.memo = {}
        self.clusters_used = set()
        self.visited = set()
        cluster = 1
        for i in range(len(self.points)):
            if i in self.visited:
                continue

            if cluster in self.clusters_used:
                cluster += 1
            self.range_query(i, eps, min, cluster)

    def fit(self, points):
        self.points = points
        self.tree = cKDTree(points)
        self.clusters = [0 * len(self.points)]
        self.memo = {}
        self.clusters_used = set()
        self.visited = set()
        self.dbscan(eps = self.distance, min = self.min_pts)

    def predict(self):
        return self.clusters


if __name__ == "__main__":
    import numpy as np
    import plotly.express as px
    import plotly
    from pathlib import Path

    np.random.seed(0)
    points = np.random.random((1500, 2))
    classifier = DBSCAN(6, 0.038)
    classifier.fit(points)
    results = classifier.predict()
    fig = px.scatter(x=points[:, 0], y=points[:, 1], color=[str(i) for i in classifier.clusters])
    fig.show()
    path = Path("plot")

    plotly.offline.plot(fig, filename=str(path))

