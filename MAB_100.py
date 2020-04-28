
# Do not edit. These are the only imports permitted.
# %matplotlib inline
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class MAB(ABC):
    """
    Abstract class that represents a multi-armed bandit (MAB)
    """

    @abstractmethod
    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}

        reward : float
            reward received from arm

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """


class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.Aa = {}
        self.ba = {}
        self.p = np.zeros(narms)
        # initialize matrix and vector list for every arms
        for index in range(narms):
            # ndims*ndims
            self.Aa[index] = np.identity(ndims)
            # ndims*1
            self.ba[index] = np.zeros((ndims, 1))

    def play(self, tround, context):
        # reshape the context from 100*1 to 10*10 matrix
        reshapedContext = context.reshape(self.narms, self.ndims)
        nextArmList = []
        # calculate p value for each arms
        for index in range(self.narms):
            xaT = reshapedContext[index]
            xa = np.transpose([xaT])
            theta = np.dot(np.linalg.inv(self.Aa[index]), self.ba[index])
            self.p[index] = np.dot(xaT, theta) + self.alpha * np.sqrt(
                np.dot(np.dot(xaT, np.linalg.inv(self.Aa[index])), xa))
        # find the arms with maximum p value
        for index, value in enumerate(self.p):
            if value == max(self.p):
                nextArmList.append(index)
        # tie-breaking
        nextArm = int(np.random.choice(nextArmList)) + 1
        return nextArm

    def update(self, arm, reward, context):
        index = arm - 1
        reshapedContext = context.reshape(self.narms, self.ndims)
        feature = np.array(reshapedContext[index])
        # change feature from (!0,) to (1,10)
        feature = feature[np.newaxis, :]
        featureT = np.transpose([reshapedContext[index]])
        # update Aa and ba
        self.Aa[index] = self.Aa[index] + np.dot(featureT, feature)
        self.ba[index] = self.ba[index] + reward * np.transpose([reshapedContext[index]])

def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    recordId = 0
    rewardList = []

    for t in range(1, nrounds + 1):
        # get next event
        armPredicted = mab.play(t, contexts[recordId])
        # find same arm as the one that was selected by logging policy
        while armPredicted != arms[recordId]:
            recordId += 1
            armPredicted = mab.play(t, contexts[recordId])
        # retain the event
        mab.update(armPredicted, rewards[recordId], contexts[recordId])
        # store the rewards in a list to evaluate the result
        rewardList.append(rewards[recordId])
        recordId += 1

    return rewardList


# Load the data file
import numpy
data = np.loadtxt('dataset.txt')
arms = data[:,0]
rewards = data[:,1]
contexts = data[:,2:]
testarms = data[9000:,0]
testrewards = data[9000:,1]
testcontexts = data[9000:,2:]

mab = LinUCB(10, 10, 1)
results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('LinUCB average reward', np.mean(results_LinUCB))


# initialize the dataset to print the figure
# figEpsGreedy = [np.mean(results_EpsGreedy[0:t]) for t in range(1, 800)]
# figUCB = [np.mean(results_UCB[0:t]) for t in range(1, 800)]
figLinUCB = [np.mean(results_LinUCB[0:t]) for t in range(1, 800)]
# use matplotlib to print the figure
# plt.plot(figEpsGreedy, 'r', linewidth=1.2, label='EpsGreedy')
# plt.plot(figUCB, 'g', linewidth=1.2, label='UCB')
plt.plot(figLinUCB, 'b', linewidth=1.2, label='LinUCB')
# label and title
plt.title('Cumulative Reward for Different MAB approach')
plt.xlabel('Round T')
plt.ylabel('per-round Cumulative Reward')
plt.legend()
plt.show()


def offlineTest(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    recordId = 0
    rewardList = []

    for t in range(1, nrounds + 1):
        # get next event
        armPredicted = mab.play(t, contexts[recordId])
        # find same arm as the one that was selected by logging policy
        if armPredicted != arms[recordId]:
            rewardList.append(0)
        # recordId += 1
        # armPredicted = mab.play(t, contexts[recordId])
        # retain the event
        # mab.update(armPredicted, rewards[recordId], contexts[recordId])
        # store the rewards in a list to evaluate the result
        else:
            rewardList.append(rewards[recordId])
        recordId += 1

    return rewardList

results_test = offlineTest(mab, testarms, testrewards, testcontexts, 1000)
print('LinUCB average reward', np.mean(results_LinUCB))
figtest = [np.mean(results_test[0:t]) for t in range(1, 1000)]
# use matplotlib to print the figure
# plt.plot(figEpsGreedy, 'r', linewidth=1.2, label='EpsGreedy')
# plt.plot(figUCB, 'g', linewidth=1.2, label='UCB')
plt.plot(figtest, 'b', linewidth=1.2, label='LinUCB')
# label and title
plt.title('Cumulative Reward for test')
plt.xlabel('Round T')
plt.ylabel('per-round Cumulative Reward')
plt.legend()
plt.show()


