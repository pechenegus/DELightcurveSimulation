
"""
DELCuse.py

Created on March  23  2014

Author: Sam Connolly

Example code for using the commands in DELCgen to simulate lightcurves

"""

from DELCgen import *
import scipy.stats as st
import numpy as np


#------- Input parameters -------

# File Route
route = ""#"/data/sdc1g08/HDData/ianCCFtest/data/"#"/route/to/your/data/"
datfile = "NGC4051.dat"#"0.5-2_part2.dat"#

# Bending power law params
A,v_bend,a_low,a_high,c = 0.03, 2.3e-4, 1.1, 2.2, 0 
# Probability density function params
kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
# Simulation params
RedNoiseL,RandomSeed,aliasTbin, tbin = 100,12,1,100 

#--------- Commands ---------------

# load data lightcurve
datalc = Load_Lightcurve(route+datfile)

# create mixture distribution to fit to PDF
mix_model = Mixture_Dist([st.gamma,st.lognorm],[3,3],[[[2],[0]],[[2],[0],]])


# estimate underlying variance of data light curve
datalc.STD_Estimate()

# simulate artificial light curve with Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                RedNoiseL,aliasTbin,RandomSeed)

# simulate artificial light curve with Emmanoulopoulos method, scipy distribution
delc_mod = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                               mix_model, (kappa, theta, lnsig, np.exp(lnmu),
                                                              weight,1-weight))

delc = datalc.Simulate_DE_Lightcurve()
  
delc.Save_Lightcurve('lightcurve.dat')
                                  
# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc,delc_mod],names=["Data LC","Timmer \& Koenig",
             "Emmanoulopoulos from model","Emmanoulopoulos from data"],bins=25)
