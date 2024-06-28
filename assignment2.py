#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):

    def sample_from_D(self, m):
        xs = np.random.uniform(0,1,m)
        ys = np.zeros(m)
        
        for i in range(m):
            if((0 <=xs[i] and xs[i] <= 0.2) or (0.4 <=xs[i] and xs[i] <= 0.6) or (0.8 <=xs[i] and xs[i] <= 1)): #If x\in I_0
                ys[i] = np.random.choice([0,1], p=[0.2,0.8])
            else:
                ys[i] = np.random.choice([0,1], p=[0.9,0.1])
        
        perm = xs.argsort()

        return np.transpose((xs[perm],ys[perm]))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        theRange = range(m_first,m_last+1,step)

        emp_errs = np.zeros(len(theRange))
        true_errs = np.zeros(len(theRange))

        for i in range(T):
            j=0

            for m in theRange:
                S = self.sample_from_D(m)

                interv ,err_cnt = intervals.find_best_interval(S[:,0],S[:,1],k)

                emp_errs[j] += float(err_cnt)/float(m)
                true_errs[j] += self.ComputeTrueErr(interv)

                j += 1
        emp_errs /= T
        true_errs /= T
        
        plt.plot(theRange, emp_errs,color="blue",label="Empirical error")
        plt.plot(theRange, true_errs,color="red",label="True error")
        plt.title("True and empirical error(s) of ERM(S_n) as a function of n")
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        return np.transpose(emp_errs,true_errs)


    #Computes the true error of a given hypothesis using the formula I derived in the PDF
    def ComputeTrueErr(self,intervals):
        SI0 = 0.6
        SI0c = 0.4

        SIcapI0 = self.ComputeAreaOfIntersect(intervals, [[0,0.2],[0.4,0.6],[0.8,1]])
        SIcapI0c = self.ComputeAreaOfIntersect(intervals, [[0.2,0.4],[0.6,0.8]])

        return 0.8*SI0 + 0.1*SI0c - 0.6*SIcapI0 + 0.8*SIcapI0c
    

    #This is a helper function that computes (as denoted in my PDF) S_{arr1 \cap arr2}
    def ComputeAreaOfIntersect(self,arr1, arr2):
        S=0

        # i and j pointers for arr1 
        # and arr2 respectively
        i = j = 0
        n = len(arr1)
        m = len(arr2)
    
        # Loop through all intervals unless one 
        # of the interval gets exhausted
        while i < n and j < m:
            
            # Left bound for intersecting segment
            l = max(arr1[i][0], arr2[j][0])
            
            # Right bound for intersecting segment
            r = min(arr1[i][1], arr2[j][1])
            
            # If segment is valid add it's probability
            if l <= r: 
                S += r-l

            # If i-th interval's right bound is 
            # smaller increment i else increment j
            if arr1[i][1] < arr2[j][1]:
                i += 1
            else:
                j += 1

        return S


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        theRange = range(k_first,k_last+1,step)

        emp_errs = np.zeros(len(theRange))
        true_errs = np.zeros(len(theRange))

        S = self.sample_from_D(m)

        i=0
        for k in theRange:
            interv ,err_cnt = intervals.find_best_interval(S[:,0],S[:,1],k)

            emp_errs[i] = float(err_cnt)/float(m)
            true_errs[i] = self.ComputeTrueErr(interv)
            print("Finished iter:",i)
            i += 1
        
        plt.plot(theRange, emp_errs,color="blue",label="Empirical error")
        plt.plot(theRange, true_errs,color="red",label="True error")
        plt.title("True and empirical error(s) of ERM(S_n) as a function of k")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        plt.show()


        return theRange[np.argmin(emp_errs)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    bestk=ass.experiment_k_range_erm(1500,1,10,1)
    print(bestk)
    '''
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
    '''

