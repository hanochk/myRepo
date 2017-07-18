def preprocess(self, point_cloud):
       numels = ((self.ranges[1, :] - self.ranges[0, :]) / self.resolutions).astype(np.int32) #Ranges 2X3  [minx miny minz ; maxx maxy maxz]
       count_view = np.zeros(numels[0:2]).astype(np.int32)
       height_view = np.ones(numels)
       ref_view = np.zeros(numels)[0:2]
       for level in range(numels[2]):
           height_view[:, :, level] = level * self.resolutions[2] + self.ranges[0, 2]
       # get indices of each point in the occupancy grid
       idxs = ((point_cloud[:, 0:3] - self.ranges[0, :]) / self.resolutions).astype(np.int32)
       condition = np.logical_and(idxs >= 0, idxs < numels).all(axis=1)
       point_cloud = point_cloud[condition, :]
       idxs = idxs[condition, :]
       for i in range(idxs.shape[0]):
           idx = idxs[i]
           count_view[idx[0], idx[1]] += 1
           if height_view[idx[0], idx[1], idx[2]] < point_cloud[i, 2]:
               height_view[idx[0], idx[1], idx[2]] = point_cloud[i, 2]
               if idx[2] == numels[2] - 1:
                   ref_view[idx[0], idx[1]] = point_cloud[3]

       return count_view, height_view, ref_view