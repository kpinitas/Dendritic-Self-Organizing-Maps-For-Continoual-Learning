#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:53:09 2020

@author: kpinitas
"""
import helper
import cupy as cp
from tqdm import tqdm
from helper import fmax,fmin


class DendSOM(object):
    """
    A class used to implement the DendSOM model

    ...

    Attributes
    ----------
    receptive_field_size : touple
        receptive field dims
    step : int
       receptive field window sdescrete step
    image_size : touble
        expected size of the input images
    num_legs : int
        the number of legs the animal has (default 4)
    d1 : int
        number of units per SOM (horizontal axis)
    d2 : int
        number of units per SOM (vertical axis)
    metric: string
            BMU identification rule
    Methods
    -------
    init_model()
        initializes the model
    a(t)
        calculates the learning rate in each time step t
    h(S,t)
        calculates the neighborhood function in each time step t
    sigma(t)
        calculates the neighborhood radius in each time step t
    train_step(t, x)
        performs a training step (for debug only)
    train_incr_task(X,y,a0,lmbd,sigma0, a_crit,r_exp)
        performs inremental task training
    train_incr_class(X,y,a0,lmbd,sigma0, a_crit,r_exp)
        performs inremental class training
    train_incr_dom(X,y,a0,lmbd,sigma0, a_crit,r_exp)
        performs inremental domain training
    train_classifier(X,y,a0,lmbd,sigma0)
        performs unsupervised classification training
    calculate_pmi()
        calsulates  PMI(BMU;label)
    classify(img=,test_mode,task)
        calculates the sum of local PMIs per label and returns the label that produces the maximum sum
    identify_bmu(vec)
        determines the BMU of each map
    model_update(bmu,vec,t)
        performs model update
       
    """   
    def __init__(self,receptive_field_size,step,image_size,d1,d2, metric='cos'):
        """
        Parameters
        ----------
        receptive_field_size : touple
            receptive field dims
        step : int
           receptive field window sdescrete step
        image_size : touble
            expected size of the input images
        num_legs : int
            the number of legs the animal has (default 4)
        d1 : int
            number of units per SOM (horizontal axis)
        d2 : int
            number of units per SOM (vertical axis)
        metric: string
            BMU identification rule
        """             
        self.image_size=image_size
        self.step=step
        self.a0 = 0.8
        self.lmbd = 1e2
        self.receptive_field_size=receptive_field_size
        self.d1=d1
        self.d2=d2
        self.sigma0 = max([d1,d2])/2
        self.model = None 
        self.metric=  metric
        self.init_model()

        
        self.S = cp.array([[[(i,j) for j in range(self.shape[2])] for i in range(self.shape[1])] for z in range(self.shape[0])])
        
    
    def init_model(self):
        """ 
        Model initialization
        
        """
        self.neurons=((self.image_size[0]-self.receptive_field_size[0])//self.step+1)*((self.image_size[1]-self.receptive_field_size[1])//self.step+1)        
        self.rec_im = (self.image_size[0]-((self.image_size[0]-self.receptive_field_size[0])%self.step),self.image_size[0]-((self.image_size[1]-self.receptive_field_size[1])%self.step))
        self.model = cp.random.uniform(size=(self.neurons, self.d1,self.d2,self.receptive_field_size[0],self.receptive_field_size[1]))
        self.shape=(self.neurons,self.d1,self.d2,self.receptive_field_size[0],self.receptive_field_size[1])



        
    def a(self, t):
        """ 
        Calculates the learning rate in each time step t
        
        Parameters
        ----------
        t: int
            time step
        lmbd: int
            controls the exponential decay
        """
        lr = self.a0*cp.exp(-t/self.lmbd) 
        return lr

    def h(self,S,t):
        """ 
        Calculates the neighborhood function in each time step t
        
        Parameters
        ----------
        t: int
            time step
        S: cupy array
            contains the position vector of each unit 
        """
        std = self.sigma(t)
        return cp.exp(-(S**2)/(2*std**2))

    def sigma(self, t):
        """
        Calculates the neighborhood radius in each time step t
        
        Parameters
        ----------
        t: int
            time step
        lmbd: int
            controls the exponential decay
        """
        sgm=self.sigma0*cp.exp(-t/self.lmbd) 
        return sgm 
    
        
    def train_step(self,t, x):
        """
        Performs a training step
        
        Parameters
        ----------
        t: int
            time step
        x: cupy array
            the input vector
        """
        sample = x
        bmu_ind = self.identify_bmu(sample)
        self.model_update(bmu_ind,sample,t)
        return t+1

   
   
    def train_incr_task(self,X,y,a0=None,lmbd=None,sigma0=None, a_crit=0.000005,r_exp = 2):
        """
        Performs inremental task training
        
        Parameters
        ----------
        X: cupy array
            train data
        y: cupy array
            train labels
        a0: fkoat
            initial learning rate
        lmbd: int
            controls the exponential decay of a and sigma
        sigma0: fkoat
            initial radius
        a_crit: float
            determins the number of iterations before a and sigma expantion
        r_exp: float
            time step penalty
        """
        t=0
        self.a0=a0 if a0!=None else self.a0
        self.sigma0=sigma0 if sigma0!=None else self.sigma0
        self.lmbd=lmbd if lmbd!=None else self.lmbd
        self.train_data = X
        self.n_classes = len(set(y.tolist()))
        self.n_tasks=self.n_classes//2
        imp_mat_shape = tuple([self.n_tasks,self.n_classes//self.n_tasks]+list(self.shape)[0:3]) 
        self.importance_matrix = cp.zeros(shape=imp_mat_shape)
        iter_crit = int(self.lmbd*cp.log(self.a0/a_crit))
        ii=0
        for cl in range(self.n_tasks):
#           draw data for task c
            print('task '+str(cl))
            c0 = y==cl*2 #class 0 for task i                
            c1= y==cl*2+1 #class 1 for task i
            c= c0+c1
            task  = X[c==True]
            task_label = (y[c==True]%2).tolist()

            for j in tqdm(range(task.shape[0])):
                ii=ii+1
                sample = task[j]
                sample=helper.create_fields(sample,self.receptive_field_size,self.step)
                label = task_label[j]
                bmu_ind = self.identify_bmu(sample)
                self.model_update(bmu_ind,sample,t)
                pred = [[cl for i in range(self.neurons)],[label for i in range(self.neurons)],[i for i in range(self.neurons)],[bmu_ind[i][0] for i in range(self.neurons)],[bmu_ind[i][1] for i in range(self.neurons)]]
                self.importance_matrix[pred]=self.importance_matrix[pred]+1
                if ii%iter_crit==0:
                    t=t//r_exp
                t=t+1
        self.mode='incr_task'
    
  
        
    def train_incr_class(self,X,y,a0=None,lmbd=None,sigma0=None,a_crit=0.000005,r_exp = 2):
        """
        Performs inremental class training
        
        Parameters
        ----------
        X: cupy array
            train data
        y: cupy array
            train labels
        a0: fkoat
            initial learning rate
        lmbd: int
            controls the exponential decay of a and sigma
        sigma0: fkoat
            initial radius
        a_crit: float
            determins the number of iterations before a and sigma expantion
        r_exp: float
            time step penalty
        """
        t=0
        self.a0=a0 if a0!=None else self.a0
        self.sigma0=sigma0 if sigma0!=None else self.sigma0
        self.lmbd=lmbd if lmbd!=None else self.lmbd
        self.train_data = X
        self.n_classes = len(set(y.tolist()))
        self.n_tasks = self.n_classes//2
        imp_mat_shape = tuple([self.n_classes]+list(self.shape)[0:3])
        self.importance_matrix = cp.zeros(shape=imp_mat_shape)
        iter_crit = int(self.lmbd*cp.log(self.a0/a_crit))
        ii=0
        for cl in range(self.n_tasks):
#           draw data for task c
            print('task '+str(cl))
            c0 = y==cl*2 #class 0 for task i                
            c1= y==cl*2+1 #class 1 for task i
            c= c0+c1
            task  = X[c==True]
            task_label = y[c==True].tolist()
            for j in tqdm(range(task.shape[0])):  
                ii=ii+1
                sample = task[j]
                sample=helper.create_fields(sample,self.receptive_field_size,self.step)
                label = task_label[j]
                bmu_ind = self.identify_bmu(sample)
                self.model_update(bmu_ind,sample,t)
                pred = [[label for i in range(self.neurons)],[i for i in range(self.neurons)],[bmu_ind[i][0] for i in range(self.neurons)],[bmu_ind[i][1] for i in range(self.neurons)]]
                self.importance_matrix[pred]=self.importance_matrix[pred]+1
                if ii%iter_crit==0:
                    t=t//r_exp
                t=t+1
        self.mode='incr_class'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_incr_dom(self,X,y,a0=None,lmbd=None,sigma0=None,a_crit=0.000005,r_exp = 2):
        """
        Performs inremental domain training
        
        Parameters
        ----------
        X: cupy array
            train data
        y: cupy array
            train labels
        a0: fkoat
            initial learning rate
        lmbd: int
            controls the exponential decay of a and sigma
        sigma0: fkoat
            initial radius
        a_crit: float
            determins the number of iterations before a and sigma expantion
        r_exp: float
            time step penalty
        """
        t=0
        self.a0=a0 if a0!=None else self.a0
        self.sigma0=sigma0 if sigma0!=None else self.sigma0
        self.lmbd=lmbd if lmbd!=None else self.lmbd
        self.train_data = X
        self.n_classes = len(set(y.tolist()))
        self.n_tasks = self.n_classes//2 
        imp_mat_shape = tuple([self.n_classes//self.n_tasks]+list(self.shape)[0:3])
        self.importance_matrix = cp.zeros(shape=imp_mat_shape)
        iter_crit = int(self.lmbd*cp.log(self.a0/a_crit))
        ii=0
        for cl in range(self.n_tasks):
#           draw data for task c
            print('task '+str(cl))
            c0 = y==cl*2 #class 0 for task i                
            c1= y==cl*2+1 #class 1 for task i
            c= c0+c1
            task  = X[c==True]
            task_label = (y[c==True]%2).tolist()
            for j in tqdm(range(task.shape[0])):
                ii=ii+1
                sample = task[j]
                sample=helper.create_fields(sample,self.receptive_field_size,self.step)
                label = task_label[j]
                bmu_ind = self.identify_bmu(sample)
                self.model_update(bmu_ind,sample,t)
                pred = [[label for i in range(self.neurons)],[i for i in range(self.neurons)],[bmu_ind[i][0] for i in range(self.neurons)],[bmu_ind[i][1] for i in range(self.neurons)]]
                self.importance_matrix[pred]=self.importance_matrix[pred]+1
                if ii%iter_crit==0:
                    t=t//r_exp
                t=t+1
                
        self.mode='incr_dom'
    
    def train_classifier(self,X,y,a0=None,lmbd=None,sigma0=None):
        """
        Performs unsupervised classification training
        
        Parameters
        ----------
        X: cupy array
            train data
        y: cupy array
            train labels
        a0: fkoat
            initial learning rate
        lmbd: int
            controls the exponential decay of a and sigma
        sigma0: fkoat
            initial radius
        """
        self.a0=a0 if a0!=None else self.a0
        self.sigma0=sigma0 if sigma0!=None else self.sigma0
        self.lmbd=lmbd if lmbd!=None else self.lmbd
        self.train_data = X
        self.n_classes = len(set(y.tolist()))
        imp_mat_shape = tuple([self.n_classes]+list(self.shape)[0:3])
        self.importance_matrix = cp.zeros(shape=imp_mat_shape)
        for t in tqdm(range(self.train_data.shape[0])):
            label = y[t].tolist()
            sample=self.train_data[t]
            sample=helper.create_fields(sample,self.receptive_field_size,self.step)
            bmu_upt = self.identify_bmu(sample)
            self.model_update(bmu_upt,sample,t)
            pred = [[label for i in range(self.neurons)],[i for i in range(self.neurons)],[bmu_upt[i][0] for i in range(self.neurons)],[bmu_upt[i][1] for i in range(self.neurons)]]
            self.importance_matrix[pred]=self.importance_matrix[pred]+1
        self.mode ='classif'
                
        
        
            
    def calculate_pmi(self):
        """
        Calsulates  PMI(BMU;label)
        """
        stability_term  = 0.0000001
        if self.mode=='incr_task':
            self.class_neur = self.importance_matrix/(cp.sum(self.importance_matrix,axis=[1])+stability_term).reshape((self.n_tasks,1,self.neurons,self.d1,self.d2))
            self.class_neur[self.class_neur==-cp.inf] = fmin
            self.class_neur[self.class_neur==cp.inf] = fmax
            self.prior  = cp.sum(self.importance_matrix,axis=[2,3,4])/cp.sum(self.importance_matrix,axis=[1,2,3,4]).reshape((self.n_tasks,1))
            self.class_neur =cp.log(self.class_neur/self.prior.reshape((self.n_tasks,self.prior.shape[1],1,1,1)))
            self.class_neur[self.class_neur==-cp.inf] = fmin
            self.class_neur[self.class_neur==cp.inf] = fmax
        else:
            self.class_neur = self.importance_matrix/(cp.sum(self.importance_matrix,axis=[0])+stability_term).reshape((1,self.neurons,self.d1,self.d2))
            self.class_neur[self.class_neur==-cp.inf] = fmin
            self.class_neur[self.class_neur==cp.inf] = fmax            
            self.prior  = cp.sum(self.importance_matrix,axis=[1,2,3])/cp.sum(self.importance_matrix)
            self.class_neur =cp.log(self.class_neur/self.prior.reshape((self.prior.shape[0],1,1,1)))
            self.class_neur[self.class_neur==-cp.inf] = fmin
            self.class_neur[self.class_neur==cp.inf] = fmax

    
    def classify(self,img=None,test_mode=True,task=None):
        """
        Calculates the sum of local PMIs per label and returns the label that produces the maximum sum
        
        Parameters
        ----------
        img: cupy array
            test sample
        test_mode: boolean
            used for debug
        task: int
            task information to be utilized
        """
        if test_mode:
            sample = helper.create_fields(img,self.receptive_field_size,self.step)
        else:
            sample=img
        
        bmu_ind = self.identify_bmu(sample)
        pmis = self.class_neur
        if type(task)!= type(None):
            pmis=pmis[task]
        infer = cp.zeros(shape=(self.neurons,pmis.shape[0]))
        for i in range(infer.shape[0]):
            for j in range(infer.shape[1]):
                infer[i,j] = pmis[j,i,bmu_ind[i][0],bmu_ind[i][1]]
        self.infer = infer        
        sum_pmis=cp.exp(cp.sum(self.infer,axis=0))
        ret = cp.argmax(sum_pmis).tolist()
        return ret
    
    
  

    def identify_bmu(self,vec):
        """
        Determines the BMU of each map
        
        Parameters
        ----------
        vec: cupy array
            input vector
        """
        if self.metric == 'cos':
            dst=cp.einsum('abcdm,aefdm->abc',self.model,vec)
            unit_mag=cp.sum(self.model**2,axis=[3,4])
            vec_mag= cp.sum(vec**2,axis=[3,4])
            denom =cp.sqrt(unit_mag *vec_mag)
            dst=dst/denom
        else:
            dst = (self.model-vec)**2
            dst = cp.sum(dst,axis=[3,4])          
            dst =cp.sqrt(dst)
            dst=cp.exp(-dst)
        d=dst
        d[d==cp.inf] = fmax
        d[d==-cp.inf] =fmin
        is_nan =cp.isnan(d)
        d[is_nan] =fmin
        max_vals = cp.max(d,axis=1)
        col_max = cp.argmax(max_vals,axis=1)
        max_vals = cp.max(d,axis=2)
        row_max = cp.argmax(max_vals,axis=1)
        
        bmus = [(row_max[i].tolist(), col_max[i].tolist()) for i in range(col_max.shape[0])]                     
        return bmus
    

        
        
    def model_update(self,bmu,vec,t):
        """
        Performs model update
        
        Parameters
        ----------
        bmu: cupy array
            maintains the positions of the selected BMUs
        vec: cupy array
            input vector
        t: int
            time step

        """
        bmu=cp.array(bmu)
        bmu=cp.expand_dims(cp.expand_dims(bmu,1),1)
        S= cp.sqrt(cp.sum((self.S-bmu)**2,3))
        S[S==cp.inf] = fmax
        dw = (vec-self.model)
        a=self.a(t) 
        h=cp.expand_dims(cp.expand_dims(self.h(S,t),3),3) 
        self.model+=a*h*dw
        
  