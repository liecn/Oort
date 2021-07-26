
class Client(object):

    def __init__(self, hostId, clientId, dis, size, speed, dropout_ratio,traces=None):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed[
'computation']
        self.bandwidth = speed[
'communication']
        self.distance = dis
        self.size = size
        self.score = dis
        self.traces = traces
        self.behavior_index = 0
        self.dropout_ratio=dropout_ratio

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def isActive(self, cur_time):
        if self.traces is None:
            return True
            
        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True

        return False

    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth)*(1-self.dropout_ratio))
        #return (3.0 * batch_size * upload_epoch*float(self.compute_speed)/1000. + model_size/float(self.bandwidth))
    def getLocalComputationTime(self, batch_size, upload_epoch, model_size):
        return 3.0 * batch_size * upload_epoch/float(self.compute_speed)