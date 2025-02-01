from sklearn.cluster import KMeans

# This class assigns players to teams based on their jersey colors using K-Means clustering.
class TeamAssigner:
    def __init__(self):
        self.team_colors = {} # Stores the two team colors(RGB Values) (cluster centers)
        self.player_team_dict = {} # Maps player IDs to their assigned team
    
    def get_clustering_model(self,image):
        # Runs K-Means clustering on an image
        
        # Reshape the image to a 2D array (each row is an RGB pixel)
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        # Extracts the dominant jersey color for an individual player
        
        # Extract the bounding box region from the frame
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Extract the upper half of the player (assumed to be the jersey)
        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get clustering model for the jersey region
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        # Determines the two team colors
        
        player_colors = []
        # Extract jersey colors for each detected player
        for _, player_detection in player_detections.items(): # _ is track_id which we are ignoring
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        # Run K-Means on all extracted player colors to find 2 team colors
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        # Store cluster centers as team colors
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0] # Team 1 color
        self.team_colors[2] = kmeans.cluster_centers_[1] # Team 2 color
        
    # The first K-Means (get_player_color) runs on individual players' jerseys.
    # The second K-Means (assign_team_color) clusters all detected player colors into two teams.


    def get_player_team(self,frame,player_bbox,player_id):
        # Determines the team for a given player
        
        # If player team already assigned, return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's dominant jersey color
        player_color = self.get_player_color(frame,player_bbox)

        # Predict the player's team based on their jersey color
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        # kmeans.predict iss using the K-Means model (trained earlier) to classify player_color into one of two clusters
        # player_color is a 1D array: [R,G,B] but K-Means expects a 2D array: [[R,G,B]] ; 1 row and 3 columns
        # kmeans.predict returns an array : [0] or [1] indicating the cluster to which the player_color belongs
        # therefore, we use [0] to get the cluster ID
        
        # K-Means labels clusters as 0 and 1.
        # But we want to represent teams as 1 and 2.
        # So we simply shift:
        team_id+=1 
        

        # Special case: Force player ID 91 into team 1
        if player_id ==91:
            team_id=1
        # Store the result
        self.player_team_dict[player_id] = team_id

        return team_id 