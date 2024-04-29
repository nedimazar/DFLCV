from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape to 2d
        image_2d = image.reshape(-1, 3)

        # Perform kmeans
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # We crop around the player
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # We get the top half since the jersey is probably there
        image = image[: image.shape[0] // 2, :]

        # Get clustering model
        kmeans = self.get_clustering_model(image)

        #  Get the cluster labels for each pixel of the image
        labels = kmeans.labels_

        # Reshape to the original shape
        clusterd_image = labels.reshape(image.shape[0], image.shape[1])

        # Get the player cluster
        corner_clusters = (
            clusterd_image[0, 0],
            clusterd_image[0, -1],
            clusterd_image[-1, -1],
            clusterd_image[-1, 0],
        )
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # This is the average player color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id
