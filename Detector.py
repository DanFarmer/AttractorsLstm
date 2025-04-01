import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from causal_ccm.causal_ccm import ccm
import scipy as sp
import itertools
import pandas as pd

class DriverDetector:
    
    def __init__(self, targetData:np.array, driverData:pd.DataFrame):
        self.targetData = targetData
        self.driverData = driverData
        self.results = None 

    ##                                         -------------  CCM  ----------------

    def __mutualInformation(self,data, delay, nBins):
        "This function calculates the mutual information given the delay"
        time_series_delayed = data[:-delay]
        time_series_original = data[delay:]

        hist_2d, _, _ = np.histogram2d(time_series_original, time_series_delayed, bins=nBins)

        joint_probs = hist_2d / np.sum(hist_2d)
        p_x = np.sum(joint_probs, axis=1)
        p_y = np.sum(joint_probs, axis=0)

        mutual_info = 0.0
        for i in range(nBins):
            for j in range(nBins):
                if joint_probs[i,j] != 0:
                    mutual_info += joint_probs[i,j] * np.log2(joint_probs[i,j] / (p_x[i] * p_y[j]))
        
        return mutual_info

    def __takensEmbedding(self,data, delay, dimension):
        "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
        if delay*dimension > len(data):
            raise NameError('Delay times dimension exceed length of data!')    
        embeddedData = np.array([data[0:len(data)-delay*dimension]])
        for i in range(1, dimension):
            embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
        return embeddedData

    def __false_nearest_neighours(self,data,delay,embeddingDimension):
        "Calculates the number of false nearest neighbours of embedding dimension"    
        embeddedData = self.__takensEmbedding(data,delay,embeddingDimension);
        #the first nearest neighbour is the data point itself, so we choose the second one
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embeddedData.transpose())
        distances, indices = nbrs.kneighbors(embeddedData.transpose())
        #two data points are nearest neighbours if their distance is smaller than the standard deviation
        epsilon = np.std(distances.flatten())
        nFalseNN = 0
        for i in range(0, len(data)-delay*(embeddingDimension+1)):
            if (0 < distances[i,1]) and (distances[i,1] < epsilon) and ( (abs(data[i+embeddingDimension*delay] - data[indices[i,1]+embeddingDimension*delay]) / distances[i,1]) > 10):
                nFalseNN += 1
        return nFalseNN

    def __isDriver(self,X,Y, verbose=False):
        if len(X) != len(Y):
            print('These time series are not the same length... X = (', len(X), ') and Y = (', len(Y),')')
            return
        
        # Determine the values for tau
        # print('Calculating the best time-delay value...')
        datDelayInfo = []
        for i in range(1,21):
            datDelayInfo.append(self.__mutualInformation(X, i, int(round(len(X)/100)))) # bin size should expect 100 points per bin...
        tau = range(1,21)[np.argmin(datDelayInfo)]
        if verbose:
            plt.figure(figsize=(10,3))
            plt.title('Mutual info to calculate the best time delay')
            plt.xlabel('tau')
            plt.ylabel('Mutual information')
            plt.plot(range(1,21), datDelayInfo, linewidth=2)
            plt.show()

        
        #print('Calculating the best embedding dimenstion...')
        # Determin the value of E
        nFNN = []
        for i in range(1,10):
            nFNN.append(self.__false_nearest_neighours(X, 1, i))

        
            E = range(1, 10)[np.where(np.array(nFNN) == 0)[0][0]]
        
        if verbose:
            plt.figure(figsize=(10,3))
            plt.title('false nearest neighbours to calculate the best embedding dimension (E)')
            plt.xlabel('E')
            plt.ylabel('Number of false nearest neighbours')
            plt.plot(range(1,10), nFNN, linewidth=2)
            plt.show()

            print('E = ', E, 'Tau = ', tau)
        
        
        L = round(len(X))
        CCM = ccm(X,Y, tau, E, L)
        if verbose:
            print('Displaying attractor mainfolds and cross-mappings...')
            CCM.visualize_cross_mapping()

            print('Displaying correlation of X -> Y ...')
            CCM.plot_ccm_correls()
        
        Xhat_My, Yhat_Mx = [], []
        L_range = range(100,len(X), 1000)
        for l in L_range:
            ccm_XY = ccm(X,Y, tau, E, l)
            ccm_YX = ccm(Y,X, tau, E, l)
            r_xy, p_xy = ccm_XY.causality()
            r_yx, p_yx = ccm_YX.causality()
            Xhat_My.append([r_xy, p_xy])
            Yhat_Mx.append([r_yx, p_yx])

        Xhat_My = np.array(Xhat_My)
        Yhat_Mx = np.array(Yhat_Mx)

        if verbose:
            print('\nPlotting prediction power of M_y and M_x against L...')
            plt.figure(figsize=(10,3))
            plt.plot(L_range, Xhat_My[:,0], label='$\hat{X}(t)|M_y$')
            plt.plot(L_range, Yhat_Mx[:,0], label='$\hat{Y}(t)|M_x$')
            plt.xlabel('L')
            plt.ylabel('correl')
            plt.legend()
            plt.show()

            print('Y -> X: personR =', np.round(Yhat_Mx[-1,0], 2), ', p_value = ', np.round(Yhat_Mx[-1,1], 4))
            if np.round(Yhat_Mx[-1,1], 4) < 0.05 :
                print('Time series Y is a driver of Series X')
            else:
                print('Time series Y is not a driver of Series X')

        return np.round(Yhat_Mx[-1,0], 2), np.round(Yhat_Mx[-1,1], 4)
    
    
    ##                                         -------------  SMAP  ----------------

    def __solve_svd(self,A,b):
        # solve B = A*C using singular value decomposition of A
        U,S,V_transpose = sp.linalg.svd(A, full_matrices=False)

        S = 1.0/S
        S_inv = np.diag(S)

        M_inverse = np.dot(np.dot(V_transpose.T, S_inv), U.T)

        return np.dot(M_inverse, b)


    def __manifold(self,X,Y):
        return np.array([X,Y]).T

    def __w(self,manifold, t_i, t_star, theta=4):
        distances = np.sqrt(np.sum((manifold - manifold[t_star,:])**2, axis=1))
        d_bar = np.mean(distances)
        return np.exp((-theta*np.sqrt(np.sum((manifold[t_i,:] - manifold[t_star,:])**2))/d_bar))

    def __B(self,manifold, t_star, P=1):
        N,_ = manifold.shape
        b = [self.__w(manifold,i,t_star)*manifold[i+P, 0] for i in np.arange(N-P)]
        return np.array(b)

    def __A(self,manifold, t_star, P=1):
        N,n = manifold.shape
        a = np.zeros(shape=(N-P, n))
        for i in np.arange(N-P):
            for j in np.arange(n):
                a[i,j] = self.__w(manifold,i,t_star)*manifold[i,j]
        return a

    def __C(self,manifold, t_star):
        return self.__solve_svd(self.__A(manifold, t_star), self.__B(manifold,t_star))
        
    def __SMap(self,X,Y, P=1):
        M = self.__manifold(X,Y)
        N,n = M.shape
        smap = np.zeros(shape=(N-P, n))
        for i in np.arange(N-P):
            smap[i,:] = self.__C(M,i)

        return np.mean(smap, axis=0)[1]
    
    def analyse(self):
        pDrivers = self.driverData.to_numpy()
        driverNames = self.driverData.columns

        _,nD = pDrivers.shape

        lD = []
        lDIndex = []

        results = pd.DataFrame(columns=driverNames, index=['Is Driver','Interation Coefficient','Pearson R', 'P Value'])

        for i in np.arange(nD):

            driver_data = pDrivers[:,i]
            pR, pV = self.__isDriver(self.targetData,driver_data)

            if pV <= 0.05 and pR > 0.8: # is a driver
                iCoeff = self.__SMap(self.targetData, driver_data) # SMAP needs to be finalised with the value of theta
                results.loc.__setitem__((('Is Driver','Interation Coefficient','Pearson R', 'P Value'),(driverNames[i])), ['Y',iCoeff,pR,pV])


            # elif pV <= 0.05 and (pR < 0.08 and pR > 0.4): # likeley part of a combination
            #     iCoeff = self.__SMap(self.targetData, driver_data) # SMAP needs to be finalised with the value of theta
            #     results.loc.__setitem__((('Is Driver','Interation Coefficient','Pearson R', 'P Value'),(driverNames[i])), ['Possible',iCoeff,pR,pV])
            #     lD.append(driver_data)
            #     lDIndex.append(i)

            else: # not a driver
                results.loc.__setitem__((('Is Driver','Interation Coefficient','Pearson R', 'P Value'),(driverNames[i])), ['N',np.NaN,pR,pV])


        lD =np.asarray(lD).T
        if len(lD.shape) >= 2:
            _,K = lD.shape
            # Try pair combinations of possible drivers
            lDNames = driverNames[lDIndex]
            for comb in itertools.combinations(np.arange(K), 2):
                driver_data = lD[:,comb[0]] * lD[:,comb[1]]
                dCombName = lDNames[comb[0]] + '*' + lDNames[comb[1]]
                results[dCombName] = [np.NaN, np.NaN, np.NaN, np.NaN]

                pR, pV = self.__isDriver(self.targetData,driver_data)

                if pV <= 0.05 and pR > 0.8: # is a driver
                    iCoeff = self.__SMap(self.targetData, driver_data) # SMAP needs to be finalised with the value of theta
                    results.loc.__setitem__((('Is Driver','Interation Coefficient','Pearson R', 'P Value'),(dCombName)), ['Y',iCoeff,pR,pV])

                else: # not a driver
                    results.loc.__setitem__((('Is Driver','Interation Coefficient','Pearson R', 'P Value'),(dCombName)), ['N',np.NaN,pR,pV])

        self.results = results
        return results
    
    def saveResults(self, path):
        self.results.to_csv(path)
        return 


    
