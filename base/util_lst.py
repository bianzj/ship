
import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import lstsq
from scipy import stats
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
import os
import shutil
import time
from util_data import *

'''
数据处理 data process: inv, fitting, analysis etc.
'''




rd = np.pi/180.0
planck_c1 = 11910.439340652
planck_c2 = 14388.291040407
temperature_threshold = 100
temperature_zero = 273.15

def doy2date(year, doy):
    '''
    doy 到日期的转换
    :param year: 年
    :param doy: day of year
    :return:  日期（月,日）
    '''
    year = np.int_(year)
    doy = np.int_(doy)
    month_leapyear = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_notleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        for i in range(0, 12):
            if doy > month_leapyear[i]:
                doy -= month_leapyear[i]
            continue
    else:
        for i in range(0, 12):
            if doy > month_notleap[i]:
                doy -= month_notleap[i]
                continue
            if doy <= month_notleap[i]:
                month = i + 1
                day = doy
                break
    return month, day

def planck(wavelength, Ts):
    '''
    基于planck公式，计算某个波长时候，从温度到辐射的转换
    :param wavelength: 波长，10.5
    :param Ts: 温度，可以是亮温，也可以是摄氏度
    :return: 辐射
    '''
    c1 = planck_c1
    c2 = planck_c2
    if isinstance(Ts * 1.0, np.float_):
        if (Ts < temperature_threshold): Ts = Ts + temperature_zero
        wavelength = np.float_(wavelength)
        Ts = np.float_(Ts)
        rad = c1 / (np.power(wavelength, 5) * (np.exp(c2 / Ts / wavelength) - 1)) * 10000
    else:
        Ts[Ts < temperature_threshold] = Ts[Ts < temperature_threshold] + temperature_zero
        wavelength = np.float_(wavelength)
        rad = c1 / (np.power(wavelength, 5) * (np.exp(c2 / Ts / wavelength) - 1)) * 10000
    return rad

def planck0(wavelength, Ts):
    '''
    基于planck公式，计算某个波长时候，从温度到辐射的转换
    :param wavelength: 波长，10.5
    :param Ts: 温度，可以是亮温，也可以是摄氏度
    :return: 辐射
    '''
    c1 = planck_c1
    c2 = planck_c2

    if (Ts < temperature_threshold): Ts = Ts + temperature_zero
    wavelength = np.float_(wavelength)
    Ts = np.float_(Ts)
    rad = c1 / (np.power(wavelength, 5) * (np.exp(c2 / Ts / wavelength) - 1)) * 10000

    return rad



def inv_planck(wavelength, rad):
    '''
    基于planck公式，计算从辐射到亮温的转换
    :param wavelength: 波长,10.5
    :param rad: 辐射
    :return: 亮温 （300）
    '''
    c1 = planck_c1 * 10000
    c2 = planck_c2
    temp = c1 / (rad * np.power((wavelength), 5)) + 1
    Ts = c2 / (wavelength * np.log(temp))
    return Ts


def vapor(T,eap):
    '''
    计算空气的水汽压
    :param T: 空气温度
    :param eap: 湿度
    :return:
    '''
    a = 7.5
    b = 237.3
    es = 6.107*np.power(10,(7.5*T/(b+T)))
    s = es*np.log(10)*a*b/np.power(b+T,2)
    ea = es*eap*0.01
    return ea



def cal_hotspot(laiWindow, vzaWindow, vaaWindow, szaWindow, saaWindow):
    G = 0.5
    rd = np.pi / 180.0
    h = 1.0
    hs = 0.05
    d = h * hs

    ind = laiWindow > 5.1
    laiWindow[ind] = 5.1

    raa = np.abs(saaWindow - vaaWindow)
    costhetanv = np.cos(vzaWindow * rd)
    costhetas = np.cos(szaWindow * rd)
    costhetavn2 = np.power(costhetanv, 2)
    costhetas2 = np.power(costhetas, 2)
    gap_n = np.exp(-G * laiWindow / costhetanv)
    sinthetas = np.sin(szaWindow * rd)
    sinthetanv = np.sin(vzaWindow * rd)
    cosalpha = costhetas * costhetanv + sinthetas * sinthetanv * np.cos(raa * rd)
    sigma_n = np.sqrt(1 / costhetas2 + 1 / costhetavn2 - 2 * cosalpha / costhetas / costhetanv)
    w_n = d / (h * sigma_n) * (1 - np.exp(-h * sigma_n / d))

    pg_n = np.exp(-(G / costhetanv + G / costhetas - w_n * np.sqrt(G * G / costhetanv / costhetas)) * laiWindow) / gap_n

    ind = (laiWindow > 0)
    return pg_n,ind


'''
反演/拟合问题
1. 最小二乘解决 Ax+b=y的线性问题；
2. fmin解决非线性问题；
3. minimize 解决非线性问题；
4. sklearn 解决线性问题；
'''



def fitting_unknown_linear_ls(y,A):
    '''
    线性反演未知数，y=ax1+bx2+cx3, x3=1
    :param y: 因变量
    :param x: 自变量
    :return:
    '''
    coeffs = np.asarray(lstsq(A, y))[0]
    return coeffs


def fitting_unknown_nonlinear_fmin(fun,y,A,xn,x0):
    '''
    非线性反演/回归，基于fmin函数，能给初值，不能给边界条件
    :param fun: 函数
    :param y: 因变量
    :param x: 自变量，是个多列数组
    :param x0: 初值
    :param xn: 列数
    :return: 回归系数
    '''
    res = fmin(fun, x0=x0, args=(y,A,xn), disp=0)
    return res



def fitting_unknown_nonlinear_min(fun,y,A,xn,x0,bnds,method = 'Powell'):
    '''
    非线性反演/回归，基于minimize函数，能给初值，也能给边界条件
    :param fun: 函数
    :param y: 因变量
    :param x: 自变量，是个多列数组
    :param x0: 初值
    :param xn: 列数
    :param bnds: 边界
    :param method: 方法，默认为powell，还有slsqp，L-BFGS-B，TNC，
    :return: 回归系数
    '''
    res = minimize(fun, x0,
                   args=(y,A,xn),
                   method=method,
                   bounds=bnds, )
    return res.x


def fitting_unknown_nonlinear_ls(fun,y,A,xn,x0,bnds,method = 'trf'):
    '''
    非线性反演/回归，基于least_squares函数，能给初值，也能给边界条件
    :param fun: 函数
    :param y: 因变量
    :param A: 矩阵
    :param x0: 初值
    :param xn: 列数
    :param bnds: 边界
    :param method: 方法，trf', 'dogbox', 'lm'
    :return: 回归系数
    '''
    res = least_squares(fun, x0,
                   args=(y,A,xn),
                   method=method,
                   bounds=bnds, )
    return res.x


def fitting_unknown_linear_sklearn(y,A,method='ridge'):
    '''
    线性反演未知数，
    :param y: 因变量
    :param x: 自变量
    :param method: 方法，推荐ridge，然后是lasso，因为lasso会倾向于拟合0
    :return: 回归系数
    '''
    alphas = [0.0001,0.001,0.01,0.1,1,10,100]
    coeffs = 1.0
    if method=='ridge':
        clf = RidgeCV(alphas=alphas, fit_intercept=False)
        clf.fit(A, y)
        coeffs = clf.coef_
    elif method =='lasso':
        lasso = LassoCV(alphas=alphas, fit_intercept=False)
        lasso.fit(A, y)
        coeffs = lasso.coef_
    elif method =='bayesianridge':
        clf = BayesianRidge(fit_intercept=False)
        clf.fit(A,y)
        coeffs = clf.coef_
    elif method =='ard':
        clf = ARDRegression(fit_intercept=False)
        clf.fit(A,y)
        coeffs = clf.coef_

    return coeffs


from scipy.sparse.linalg import lsqr
def fitting_unknown_linear_sparse(y,A,iter=100):
    '''
    找到大型稀疏线性方程组的least-squares 解。
    :param y:
    :param x:
    :param iter:
    :return:
    '''
    coeffs = lsqr(A,y,iter_lim=iter)[0]
    return coeffs




