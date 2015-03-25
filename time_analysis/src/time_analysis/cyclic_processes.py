#!/usr/bin/env python

from __future__ import division
import sys
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import vonmises
from sklearn.mixture import GMM

import directional_statistics as ds

class vmm_model:
    def __init__(self,vec,xfit,c,period):
        self.c = c
        [mu,kappa,weights,yfit,comp_fit,self.bic] = mvm_func.mvm(list(to_rad(vec)),list(to_rad(xfit)),c)
        if mu[0] != 0:
            sort_idx = np.argsort(from_rad(mu,period))
            self.mu = from_rad(mu,period)[sort_idx]
            self.kappa = np.asarray(kappa)[sort_idx]
            self.weights = np.asarray(weights)[sort_idx]
            self.yfit = list(yfit)
            self.comp_fit = [list(x) for x in np.asarray(comp_fit)[sort_idx]]
        else:
            self.mu = []
            self.kappa = []
            self.weights = []
            self.yfit = []
            self.comp_fit = []

class activity_time:
    def __init__(self,raw,period=86400.0,interval=1800.0):
        self.period = period
        self.interval = interval
        self.raw = raw
        self.vec, self.ind = time_wrap(raw)
        self.shifted_vec, self.shifted_0 = best_cut(self.vec)
        
        self.circ_mu = ds.circ_mean(self.vec,high=self.period,low=0)
        self.circ_sd = ds.circ_std(self.vec,high=self.period,low=0)
        self.kappa = ds.kappa(self.vec,high=self.period,low=0)
    
        self.n_bins = int(self.period/self.interval)
        self.bins = binning(self.vec,self.n_bins,self.interval)
        self.xfit = np.arange(0,period+1,300)
    
        self.models, self.aic, self.bic = gmm_numpy(self.shifted_vec,5)
        best = np.argmin(self.bic)

        self.shifted_xfit = np.arange(self.shifted_0,self.shifted_0+self.period,300)
        logprob, responsibilities = self.models[best].score_samples(normalize(self.shifted_xfit))
        pdf = np.exp(logprob)

        scale_factor = (sum(self.bins)/self.n_bins) / (sum(pdf)/(self.period/300))
        yfit = [y * scale_factor for y in pdf]
        yfit = np.roll(yfit,int(self.shifted_0/(self.period/300)+1))
        self.yfit_mm = np.append(yfit,yfit[0])
        self.comp_fit_mm = []

        all_pdf_log, resp = self.models[best].score_samples(normalize(np.arange(0,period,1)))
        self.max_pd = max(np.exp(all_pdf_log))

        self.cluster_set = clustering(list(self.shifted_vec),5,self.period,self.interval)

    def query_model(self,x):
        best = np.argmin(self.bic)
        logprob, resp = self.models[best].score_samples(normalize([x]))
        return np.exp(logprob[0])/self.max_pd

            
    def display_indexes(self,plot_options,clusters,save_file):

        print '\ncircular mean:  ', sec2time(self.circ_mu), u'\u00B1', sec2time(self.circ_sd)
        print '\n kappa: ', "%0.2f" % self.kappa

        best_m = np.argmin(self.bic)
        m = self.models[best_m]
        print '\nbest fitting obtained with ', best_m+1, ' distributions'
        if best_m > 0:
            print '\nmeans: ', [sec2time(x) for x in m.mu]
            #            print 'std  : ', [sec2time(self.period/(x**2)) for x in m.kappa]
            print 'kappa: ', ["%0.2f" %x for x in m.kappa]
            print 'wgts : ', ["%0.2f" %x for x in m.weights]
            #        self.clusters.display_time_cluster_indexes()
        stop = self.plot_fitting(plot_options,clusters,save_file)
        plt.close('all')
        return stop

    def plot_fitting(self,plot_options,clusters,save_file):
        
        [title_text, hist_col, curve_col] = plot_options
        n_bins = int(self.period/self.interval)
        n_hours = self.period/(60*60.0)

        ### polar plot
        ax1 = radial_time_axes(1,2,1)
 #        ax1.hist(to_rad(self.vec,self.period),n_bins,range=[0, 2*np.pi],color=hist_col,alpha=0.75)
        ax1.bar(to_rad(np.arange(0,self.period,self.interval)),np.sqrt(self.bins[:-1]),width=to_rad(self.interval),color=hist_col,alpha=0.75)
        ax1.plot(to_rad(self.xfit,self.period),[np.sqrt(x) for x in self.yfit_mm],color=curve_col)
        plt.title(title_text, y=1.08)
 #        plt.show()
        if clusters:
            ax_limits = ax1.axis()
            mu_plot = to_rad(clusters.C)
            sd_plot = to_rad(clusters.L)
    #        breaks = time_wrap(self.end_breaks)
            for c in range(clusters.K):
                shade = colorsys.hsv_to_rgb(175, float(clusters.T[c]-min(clusters.T))/(max(clusters.T)-min(clusters.T))*0.75+0.25,1)
                ax1.plot(np.linspace((mu_plot[c]-sd_plot[c]),(mu_plot[c]+sd_plot[c]),int(sd_plot[c]*100)),np.ones(int(sd_plot[c]*100))*ax_limits[3]*0.98,color=shade,linewidth=5)
 #                arc_level = ax_limits[3]*float(clusters.T[c])/max(clusters.T)*0.98
 #                ax1.plot(np.linspace((mu_plot[c]-sd_plot[c]),(mu_plot[c]+sd_plot[c]),int(sd_plot[c]*100)),np.ones(int(sd_plot[c]*100))*arc_level,color='b',linewidth=2)

        ### linear plot
        ax3 = plt.subplot(1,2,2)
        ax3.hist([t/(60*60.0) for t in self.vec],n_bins,range=[0,n_hours],color=hist_col,alpha=0.75)
        plt.xlim([0,n_hours])
        plt.gca().set_ylim(bottom=0)
        plt.xlabel('Time [h]')
        plt.ylabel('Begin/end time frequency', color='darkgreen')
        ax3.set_xticks(np.arange(0,n_hours))
        for comp in self.comp_fit_mm:
            ax3.plot([t/(60*60.0) for t in self.xfit],comp,color='c',linewidth=2)
        ax3.plot([t/(60*60.0) for t in self.xfit],self.yfit_mm,color=curve_col,linewidth=2)
            
        plt.title(title_text)

        if save_file:
            figure = plt.gcf() # get current figure
            figure.set_size_inches(16, 8)
            plt.savefig(save_file+'.jpg', bbox_inches='tight', dpi = 100)
            plt.close()
            return False
        else:
            plt.show()
            click = plt.waitforbuttonpress() # mouse click continues, keyboard key stops
            return click

########################

class dynamic_clusters:
    def __init__(self):
        self.K = 0
        self.N = []
        self.LS = []
        self.SS = []
        self.C = []
        self.R = []
        self.D = []
        self.T = []
        self.L = []

    def add_cluster(self,d,t):
        self.K += 1
        self.N.append(1)
        self.T.append(d)
        self.LS.append(t)
        self.SS.append(t**2)
        self.C.append(t)
        self.R.append(900.0) # 15 minutes, maximum distance for joininig a 1 element cluster
        self.D.append(1800.0)
        self.L.append(900.0)
        
    def update_cluster(self,i,d,t):
        self.N[i] += 1
        self.T[i] = d # timestamp is age of newest element
        self.LS[i] += t
        self.SS[i] += t**2
        self.compute_measures(i)
        self.L[i] = self.R[i]*np.sqrt(1+d/self.T[i])
        
    def add_element(self,d,t):
        if self.K == 0:
            self.add_cluster(d,t)
        else:
            i = np.argmin([abs(c-t) for c in self.C])
            if abs(self.C[i]-t) > 2*self.L[i]:
                self.add_cluster(d,t)
            else:
                self.update_cluster(i,d,t)
        self.update_record(d,t)
    
    def update_record(self,d,t):
        for c in range(self.K):
            self.L[c] = self.R[c]*np.sqrt(1+d/self.T[c])
        seq = sorted(range(self.K), key=lambda i: self.C[i])
        c = 0
        while c < len(seq)-1:
            if self.C[seq[c]] + self.L[seq[c]] > self.C[seq[c+1]] - self.L[seq[c+1]]:
                self.cluster_merge(seq[c],seq[c+1],d,t) # merge overlapping clusters
                seq = sorted(range(self.K), key=lambda i: self.C[i])
            c += 1
                   
    def cluster_merge(self,i,j,d,t):
        self.N[i] += self.N[j]
        self.T[i] = np.max([self.T[i], self.T[j]])
        self.LS[i] += self.LS[j]
        self.SS[i] += self.SS[j]
        self.compute_measures(i)
        self.L[i] = self.R[i]*np.sqrt(1+d/self.T[i])
        self.delete_cluster(j)
        
    def compute_measures(self,i):
        self.C[i] = self.LS[i]/self.N[i]
        self.R[i] = np.min([np.sqrt(self.SS[i]/self.N[i] - (self.LS[i]/self.N[i])**2),1800]) # limit radius to 20mins
        self.D[i] = np.min([np.sqrt((2*self.N[i]*self.SS[i]-2*self.LS[i]**2)/(self.N[i]*(self.N[i] - 1))),1800])

    def delete_cluster(self,i):
        self.K -= 1
        del self.N[i]
        del self.T[i]
        del self.LS[i]
        del self.SS[i]
        del self.C[i]
        del self.R[i]
        del self.D[i]
        del self.L[i]
       
    def query_clusters(self,t):
        p = np.min([abs(c-t)/(2*l) for (c,l) in zip(self.C,self.L)])
        return (1-p)
         
#################################  

def clustering(data_vec,max_c,period,interval):    
    n = len(data_vec)
    max_clusters = np.min([max_c+1,n])
    cl_breaks = []
    cl_means = []
    cl_std = []
    cl_circmean = []
    cl_circstd = []
    cl_kappa = []
    cl_mode = []
    cl_maxy = []
    cl_weights = []
    
    for cl in range(2,max_clusters):
        breaks = best_split(data_vec, cl)
        c = len(breaks)
        class_mean = []
        class_std = []
        class_count = []
        class_circ_mu = []
        class_circ_std = []
        class_kappa = []
        class_mode = []
        class_maxy = []
        class_wgt = []
        
        for i in range(c-1):
            if breaks[i] == 0:
                class_start = 0
            else:
                class_start = data_vec.index(breaks[i]) + 1
            class_end = data_vec.index(breaks[i+1])
            class_vec = data_vec[class_start:class_end+1]
            class_mean.append(np.mean(class_vec) % period)
            class_std.append(np.std(class_vec))
            class_count.append(class_end-class_start+1)
            class_circ_mu.append(ds.circ_mean(class_vec,0,period))
            class_circ_std.append(ds.circ_std(class_vec,0,period))
            class_kappa.append(ds.kappa(class_vec,0,period))
            bin_size = int(period/interval)
            class_bins = binning([c % period for c in class_vec],bin_size,interval)
            class_mode.append(np.argmax(class_bins)*bin_size + bin_size/2)
            class_maxy.append(np.max(class_bins))
            class_wgt.append(class_count[-1]/float(n))
            
        cl_breaks.append([b % period for b in breaks[1:]])
        cl_means.append(class_mean)
        cl_std.append(class_std)
        cl_circmean.append(class_circ_mu)
        cl_circstd.append(class_circ_std)
        cl_kappa.append(class_kappa)
        cl_mode.append(class_mode)
        cl_maxy.append(class_maxy)
        cl_weights.append(class_wgt)
        
    clusters = cluster_set(cl_breaks, cl_means, cl_std, cl_circmean, cl_circstd, cl_kappa, cl_mode, cl_maxy, cl_weights)
    return clusters
 
class cluster_set:
    def __init__(self,br,mu,sd,cmu,csd,k,mo,my,wgt):
        self.breaks = br
        self.mu = mu
        self.std = sd
        self.cmu = cmu
        self.cstd = csd
        self.kappa = k
        self.mode = mo
        self.maxy = my
        self.wgt = wgt
 #        self.cluster_fitting()

    def display_time_cluster_indexes(self):
        n_sets = len(self.breaks)
        for cl_set in range(n_sets):
            print cl_set + 2, '\n clusters:'
            print ' brk:     ', [sec2time(x) for x in self.breaks[cl_set]]
            print '  mu:     ', [sec2time(x) for x in self.mu[cl_set]]
            print 'mode:     ', [sec2time(x) for x in self.mode[cl_set]]
            print ' std:     ', [sec2time(x) for x in self.std[cl_set]]
            print 'c mu:     ', [sec2time(x) for x in self.cmu[cl_set]]
            print 'cstd:     ', [sec2time(x) for x in self.cstd[cl_set]]
            print '   k:     ', ["%0.2f" %x for x in self.kappa[cl_set]]
            print ' wgt:     ', ["%0.2f" %x for x in self.wgt[cl_set]]

    def cluster_fitting(self):
        n_sets = len(self.breaks)
        xfit = np.arange(0,self.period+1,self.interval)
        self.yfit = []
        for cl_set in range(n_sets):
            yfit_cl = []
            for c in range(cl_set+2):
                new_fit = fit_curve(to_rad(xfit,self.period),'v',[self.maxy[cl_set][c],self.kappa[cl_set][c],to_rad(self.mode[cl_set][c],self.period),1])
                yfit_cl.append(new_fit)
            self.yfit.append(yfit_cl)

##################################
      
def best_split(dataList,numClass):
# implementation of Jenks Natural Breaks algorithm
    dataList.sort()
    k = len(dataList)
    mat1 = [range(numClass+1) for i in range(k+1)]
    mat2 = [range(numClass+1) for i in range(k+1)]
    for i in range(1,numClass+1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2,k+1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2,k+1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1,l+1):
            i3 = l - m + 1
            val = float(dataList[i3-1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2,numClass+1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    kclass = range(numClass+1)
    kclass[numClass] = float(dataList[k - 1])
    for countNum in range(numClass,1,-1):
        rank = mat1[k][countNum]
        kclass[countNum - 1] = dataList[rank - 2]
        k = rank - 1
    return kclass#, classMean, classStd, classCount, classCircMu, classCircStd
  
####################################
  
def fit_curve(xfit,fit_type,param):
    if fit_type == 'n':
        yfit = mlab.normpdf(xfit ,param[1], param[2]) # mu, sigma
    elif fit_type == 'v':
        yfit = vonmises.pdf(xfit, kappa=param[1], loc=param[2], scale=param[3]) # kappa, circ_mu, circ_sd
    return yfit
    
def gmm_numpy(vec,max_C):
    n = len(vec)
    N = min(n,max_C)
    X = np.array(normalize(vec))
    
    models = []
    for i in range(N):
        try:
            models.append(GMM(i+1).fit(X))
        except RuntimeError:
            print "Fitting failed, cluster n. ", str(i+1)
    
    aic = [m.aic(X) for m in models]
    bic = [m.bic(X) for m in models]
    return models, aic, bic

###################################

def best_cut(vec,period=86400.0):
    largest = vec[0] - vec[-1] + period
    i_shift = 0
    cut = ((vec[0]  + period + vec[-1])/2)
    for i in range(len(vec)-1):
        dist = vec[i+1]-vec[i]
        if dist > largest:
            largest = dist
            i_shift = i+1
            cut = (vec[i+1]+vec[i])/2
    if cut >= vec[-1]:
        cut = cut - period
    shifted = np.append(vec[i_shift:],[t + period for t in vec[:i_shift]])
    return shifted, cut

def sliding_mean(original_vec,period=86400.0):
    data_vec = original_vec[:]
    n = len(data_vec)
    fmu = np.mean(data_vec)
    fstd = np.std(data_vec)
    shifted_vec = [x for x in data_vec]
    timeshift = ((data_vec[0]  + period + data_vec[-1])/2)
    if timeshift > data_vec[0]:
        timeshift = timeshift - period
    for shift in range(n-1):
        data_vec[shift] = data_vec[shift] + period #[x+86400 if x<=data_vec[shift] else x for x in data_vec]
        shift_std = np.std(data_vec)
        if shift_std < fstd - 0.1: # tolerance for np.std drift error
            fstd = shift_std
            fmu = np.mean(data_vec) % period
            shifted_vec = data_vec[:]
            timeshift = (data_vec[shift+1] + data_vec[shift] - period)/2
    return fmu,fstd,sorted(shifted_vec),timeshift
 
#################################
   
def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)
    return ax2

def radial_time_axes(a,b,c):
    ax = plt.subplot(a,b,c,polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/12))
    ax.set_xticklabels([str(sh)+':00' for sh in range(24)])
    return ax        

##################################
   
def time2sec(time_string):
    time_sec = int(time_string.split(':')[0])*(60*60) + int(time_string.split(':')[1])*60 + int(time_string.split(':')[2][:time_string.split(':')[2].find('.')])
    return time_sec

def sec2time(time_sec):
    if time_sec < 0:
        time_sec += 24*3600
    hrs = time_sec // 3600
    time_sec %= 3600
    mins = time_sec // 60
    secs = time_sec % 60
    return "%02i:%02i:%02i" % (hrs, mins, secs)
    
def time_wrap(time_vec,period=86400):
    wrapped_vec = [t % period for t in time_vec]
    ind = np.argsort(wrapped_vec)
    return sorted(wrapped_vec), ind

def binning(vec,n_bins,interval=1800):
    bins = [0]*n_bins
    bin_vec = [int(t/interval) for t in vec]
    for h in bin_vec:
        bins[h] += 1
    bins.append(bins[0])
    return bins

def normalize(vec,period=86400.0):
    return [x*2 / period - 1.0 for x in vec]    

def denormalize(vec,period=86400.0):
    return [((x + 1.0) * period/2) % period for x in vec]

def within_sigma(x,mu,sigma,k):
    return (x > mu - sigma * k and x < mu + sigma * k)
    
def to_rad(vec,period=86400.0):
    return np.asarray(vec)*2*np.pi/period
 
def from_rad(vec,period=86400.0):
    return np.asarray(vec)*period/(2*np.pi)%period
   
