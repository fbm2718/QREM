"""
Created on 08.03.2021

@author: fbm
@contact: filip.b.maciejewski@gmail.com
"""
import qutip
import numpy as np
import picos as pic
import cvxopt as cvx
import cmath
import time
import os
import pickle
from tqdm import tqdm
import scipy as sc
import math
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import qiskit as qkt

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

'''
Our library
'''
from povms_qi import ancillary_functions as anf
from povms_qi import povmtools
from povms_qi.povms_implementation.simulation import PM_simulability_tools as pm_sim
from povms_qi.state_discrimination import state_discrimination_sdp as sd_sdp
from povms_qi.state_discrimination import state_discrimination_functions as sd_anc
from povms_qi import fixing_stats as fs
from povms_qi.state_discrimination import pbt_functions as pbt_fun
from povms_qi import povm_data_tools as pdt
from povms_qi import data_tools as dt

from VQE_mitigation.c_QAOA import functions_QAOA as QAOA_fun
from VQE_mitigation.a_GENERAL import functions_probs as fp
from VQE_mitigation.c_QAOA import functions_rectangles as fr
import scipy.optimize as scopt
from VQE_mitigation.a_GENERAL import SAT2_functions as sat2_fun

from VQE_mitigation.c_QAOA.QAOA_sampling_abstract import functions_now_QAOA as main_fun
import Flavio_SDP_functions as F_SDP