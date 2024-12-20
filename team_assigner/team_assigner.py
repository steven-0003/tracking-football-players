from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self) -> None:
        self.team_colours = {}
        self.player_team = {}

    def assign_team_colour(self, frame, player_detections):
        player_colours = []

        for _,detection in player_detections.items():
            bb = detection['bbox']
            colour = self.get_player_colour(frame, bb)
            player_colours.append(colour)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colours)

        self.kmeans = kmeans

        self.team_colours[0] = kmeans.cluster_centers_[0]
        self.team_colours[1] = kmeans.cluster_centers_[1]

    def get_player_colour(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Take the top half of the image
        top_half = img[0:int(img.shape[0]/2),:]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half)

        # Get cluster labels
        labels = kmeans.labels_
        clustered_img = labels.reshape(top_half.shape[0],top_half.shape[1])

        # Get player cluster
        corner_clusters = [clustered_img[0,0], clustered_img[0,-1], clustered_img[-1,0], clustered_img[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def get_clustering_model(self, img):
        img_2d = img.reshape(-1,3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(img_2d)

        return kmeans
    
    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        player_colour = self.get_player_colour(frame, bbox)
        team_id = self.kmeans.predict(player_colour.reshape(1,-1))[0]

        self.player_team[player_id] = team_id
        
        return team_id