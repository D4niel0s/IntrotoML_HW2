#################################
# Your name: Daniel Volkov
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
        
        plt.figure(1)
        plt.plot(theRange, emp_errs,color="blue",label="Empirical error")
        plt.plot(theRange, true_errs,color="red",label="True error")
        plt.title("True and empirical error(s) of ERM(S_n) as a function of n")
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        return np.transpose((emp_errs,true_errs))


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

            i += 1
        
        plt.figure(2)
        plt.plot(theRange, emp_errs,color="blue",label="Empirical error")
        plt.plot(theRange, true_errs,color="red",label="True error")
        plt.title("True and empirical error(s) of ERM(S_n) as a function of k")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        return theRange[np.argmin(emp_errs)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        theRange = np.array(range(k_first,k_last+1,step))
        deltak = (0.1/np.power(theRange,2))
        penalty = 2*np.sqrt((2*theRange + np.log((2)/(deltak)))/(m))
        

        emp_errs = np.zeros(len(theRange))
        true_errs = np.zeros(len(theRange))

        S = self.sample_from_D(m)

        i=0
        for k in theRange:
            interv ,err_cnt = intervals.find_best_interval(S[:,0],S[:,1],k)

            emp_errs[i] = float(err_cnt)/float(m)
            true_errs[i] = self.ComputeTrueErr(interv)

            i += 1
        
        plt.figure(3)
        plt.plot(theRange, emp_errs,color="blue",label="Empirical error")
        plt.plot(theRange, true_errs,color="red",label="True error")
        plt.plot(theRange, penalty,color="orange",label="Penalty")
        plt.plot(theRange, penalty+emp_errs,color="green",label="Penalty + Emp. error")
        plt.title("SRM penalty + empirical error, with true and emp. errors")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        return theRange[np.argmin(penalty+emp_errs)]

    def cross_validation(self, m):
        train_set = self.sample_from_D(int(0.8*m))
        holdout_set = self.sample_from_D(int(0.2*m))

        models = [0 for i in range(1,11)]
        holdout_errs = np.zeros(10)
        for k in range(1,11):
            models[k-1], _ = intervals.find_best_interval(train_set[:,0],train_set[:,1],k)
        for i in range(1,11):
            holdout_errs[i-1] = self.ComputeEmpErr(models[i-1], holdout_set)

        plt.figure(4)
        plt.plot(np.arange(1,11), holdout_errs, color="red", label="Validation error")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.title("Validation error as a function of k")
        plt.legend()
        plt.show()

        return models[np.argmin(holdout_errs)], np.argmin(holdout_errs)+1

    #################################
    # Place for additional methods


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

    #Computes the empirical error of a given hypothesis on a given dataset
    def ComputeEmpErr(self,interv, set):
        err_cnt = 0

        for i in range(len(set)):
            if(self.classify(interv, set[i][0]) != set[i][1]):
                err_cnt += 1

        return float(err_cnt)/float(len(set))

    #Classifies a given sample with the hypothesis given by interv
    def classify(self,interv, sample):
        for i in interv:
            if (i[0]<=sample and sample <= i[1]):
                return 1
        return 0
        



    #################################

if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
