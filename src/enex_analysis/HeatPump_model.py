import numpy as np
import math
# from . import calc_util as cu
import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import erf
import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import minimize

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]

sigma = 5.67*10**-8 # Stefan-Boltzmann constant [W/m²K⁴]

# https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462 # direct solar entropy coefficient [-]
k_d = 0.0014 # diffuse solar entropy coefficient [-]

# Shukuya - Exergy theory and applications in the built environment, 2013
# The ratio of chemical exergy to higher heating value of liquefied natural gas (LNG) is 0.93.
ex_eff_NG   = 0.93 # exergy efficiency of natural gas [-]

SP = np.sqrt(np.pi) # Square root of pi

#%%
# function
def darcy_friction_factor(Re, e_d):
    '''
    Calculate the Darcy friction factor for given Reynolds number and relative roughness.
    
    Parameters:
    Re (float): Reynolds number
    e_d (float): Relative roughness (e/D)
    
    Returns:
    float: Darcy friction factor
    '''
    # Laminar flow
    if Re < 2300:
        return 64 / Re
    # Turbulent flow
    else:
        return 0.25 / (math.log10(e_d / 3.7 + 5.74 / Re ** 0.9)) ** 2

def calc_h_vertical_plate(T_s, T_inf, L):
    '''
    📌 Function: compute_natural_convection_h_cp
    이 함수는 자연 대류에 의한 열전달 계수를 계산합니다.
    🔹 Parameters
        - T_s (float): 표면 온도 [K]
        - T_inf (float): 유체 온도 [K]
        - L (float): 특성 길이 [m]
    🔹 Return
        - h_cp (float): 열전달 계수 [W/m²K]
    🔹 Example
        ```
        h_cp = compute_natural_convection_h_cp(T_s, T_inf, L)
        ```
    🔹 Note
        - 이 함수는 자연 대류에 의한 열전달 계수를 계산하는 데 사용됩니다.
        - L은 특성 길이로, 일반적으로 물체의 길이나 직경을 사용합니다.
        - 이 함수는 Churchill & Chu 식을 사용하여 열전달 계수를 계산합니다.
    '''
    # 공기 물성치 @ 40°C
    nu = 1.6e-5  # 0.000016 m²/s
    k_air = 0.027 # W/m·K
    Pr = 0.7 # Prandtl number 
    beta = 1 / ((T_s + T_inf)/2) # 1/K
    g = 9.81 # m/s²

    # Rayleigh 수 계산
    delta_T = T_s - T_inf
    Ra_L = g * beta * delta_T * L**3 / (nu**2) * Pr

    # Churchill & Chu 식 https://doi.org/10.1016/0017-9310(75)90243-4
    Nu_L = (0.825 + (0.387 * Ra_L**(1/6)) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
    h_cp = Nu_L * k_air / L  # [W/m²K]
    
    return h_cp

def linear_function(x, a, b):
    return a * x + b

def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def quartic_function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def print_balance(balance, decimal=2):
    '''
    📌 Function: print_balance

    이 함수는 주어진 balance 딕셔너리를 이용하여 Energy, Entropy, Exergy balance를 출력합니다.

    🔹 Parameters:
        - balance (dict): Energy, Entropy, Exergy balance 딕셔너리
        - decimal (int, optional): 소수점 이하 출력 자릿수 (기본값: 2)

    🔹 Returns:
        - None (출력만 수행)

    🔹 출력 형식:
        - 서브시스템 별 balance 정보 출력
        - in, out, consumed, generated 등의 카테고리를 구분하여 출력
        - 각 값은 지정된 소수점 자릿수까지 반올림하여 표시

    🔹 Example:
        ```
        print_balance(exergy_balance, decimal=2)
        ```

    🔹 실행 예시:
        ```
        HOT WATER TANK EXERGY BALANCE: =====================

        IN ENTRIES:
        $X_{w,comb,out}$: 5000.00 [W]

        OUT ENTRIES:
        $X_{w,tank}$: 4500.00 [W]
        $X_{l,tank}$: 400.00 [W]

        CONSUMED ENTRIES:
        $X_{c,tank}$: 100.00 [W]

        GENERATED ENTRIES:
        $S_{g,tank}$: 50.00 [W/K]
        ```
    '''
    total_length = 50
    
    balance_type = "energy"
    unit = "[W]"
    
    for subsystem, category_dict in balance.items(): 
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            if "gen" in category:
                balance_type = "entropy"
                unit = "[W/K]"
            elif "con" in category:
                balance_type = "exergy"
    
    for subsystem, category_dict in balance.items(): 
        # subsystem: hot water tank, mixing valve...
        # category_dict: {in: {a,b}, out: {a,b}...} 
        text = f"{subsystem.upper()} {balance_type.upper()} BALANCE:"
        print(f'\n\n{text}'+'='*(total_length-len(text)))
        
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            # terms: {a,b}
            # a,b..: symbol: value
            print(f"\n{category.upper()} ENTRIES:")
            
            for symbol, value in terms.items():
                print(f"{symbol}: {round(value, decimal)} {unit}")

def calculate_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_ref_int, Q_ref_max, COP_ref):
    '''
    https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in cooling mode.

    Parameters:
    - T_a_int_out : Indoor air temperature [K]
    - T_a_ext_in  : Outdoor air temperature [K]
    - Q_ref_int     : Indoor heat load [W]
    - Q_ref_max     : Maximum cooling capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    - EIR : Energy input to cooling output ratio
    - COP_ref : the reference COP at the standard conditions
    '''
    PLR = Q_ref_int / Q_ref_max
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP

def calculate_ASHP_heating_COP(T0, Q_ref_int, Q_ref_max):
    '''
    https://www.mdpi.com/2071-1050/15/3/1880
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in heating mode.

    Parameters:
    - T0 : Enviromnetal temperature [K]
    - Q_ref_int : Indoor heat load [W]
    - Q_ref_max : Maximum heating capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    '''
    PLR = Q_ref_int / Q_ref_max
    COP = -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477)**2 + 0.0941 * cu.K2C(T0) + 4.34
    return COP

def calculate_GSHP_COP(Tg, T_cond, T_evap, theta_hat):
    """
    https://www.sciencedirect.com/science/article/pii/S0360544219304347?via%3Dihub
    Calculate the Carnot-based COP of a GSHP system using the modified formula:
    COP = 1 / (1 - T0/T_cond + ΔT * θ̂ / T_cond)

    Parameters:
    - Tg: Undisturbed ground temperature [K]
    - T_cond: Condenser refrigerant temperature [K]
    - T_evap: Evaporator refrigerant temperature [K]
    - theta_hat: θ̂(x0, k_sb), dimensionless average fluid temperature -> 논문 Fig 8 참조, Table 1 참조

    Returns:
    - COP_carnot_modified: Modified Carnot-based COP (float)
    """

    # Temperature difference (ΔT = T0 - T1)
    if T_cond <= T_evap:
        raise ValueError("T_cond must be greater than T_evap for a valid COP calculation.")
    
    delta_T = Tg - T_evap

    # Compute COP using the modified Carnot expression
    denominator = 1 - (Tg / T_cond) + (delta_T /(T_cond*theta_hat))

    if denominator <= 0:
        return float('nan')  # Avoid division by zero or negative COP

    COP = 1 / denominator
    return COP

def f(x):
    return x*erf(x) - (1-np.exp(-x**2))/SP

def chi(s, rb, H, z0=0):
    h = H * s
    d = z0 * s
    
    temp = np.exp(-(rb*s)**2) / (h * s)
    Is = 2*f(h) + 2*f(h+2*d) - f(2*h+2*d) - f(2*d)
    
    return temp * Is

_g_func_cache = {}
def G_FLS(t, ks, as_, rb, H):
    key = (round(t, 0), round(ks, 2), round(as_, 6), round(rb, 2), round(H, 0))
    if key in _g_func_cache:
        return _g_func_cache[key]

    factor = 1 / (4 * np.pi * ks)
    
    lbs = 1 / np.sqrt(4*as_*t)
    
    # Scalar 값인 경우 shape == (,).
    single = len(lbs.shape) == 0
    # 0차원에 1차원으로 변경.
    lbs = lbs.reshape(-1)
        
    # 0 부터 inf 까지의 적분값 미리 계산.
    total = integrate.quad(chi, 0, np.inf, args=(rb, H))[0]
    # ODE 초기값.
    first = integrate.quad(chi, 0, lbs[0], args=(rb, H))[0]
   
    # Scipy의 ODE solver의 인자의 함수 형태는 dydx = f(y, x).
    def func(y, s):
        return chi(s, rb, H, z0=0)
    
    values = total - integrate.odeint(func, first, lbs)[:, 0]
    
    # Single time 값은 첫 번째 값만 선택하여 float를 리턴하도록 함.
    if single:
        values = values[0]

    result = factor * values
    _g_func_cache[key] = result
    return result

def generate_balance_dict(subsystem_category):
    energy_balance = {}; entropy_balance = {}; exergy_balance = {}
    energy_balance_category = ['in', 'out']
    entropy_balance_category = ['in', 'gen','out']
    exergy_balance_category = ['in', 'con','out']
   
    for subsystem in subsystem_category:
        energy_balance[subsystem] = {}
        for category in energy_balance_category:
            energy_balance[subsystem][category] = {}

    for subsystem in subsystem_category:
        entropy_balance[subsystem] = {}
        for category in entropy_balance_category:
            entropy_balance[subsystem][category] = {}
            
    for subsystem in subsystem_category:
        exergy_balance[subsystem] = {}
        for category in exergy_balance_category:
            exergy_balance[subsystem][category] = {}
            
    return energy_balance, entropy_balance, exergy_balance

def generate_entropy_exergy_term(energy_term, Tsys, T0, fluid = None):
    """
    Calculates the entropy and exergy terms based on the provided energy term and temperatures.
    Parameters:
        energy_term (float): The energy value for which entropy and exergy are to be calculated.
        Tsys (float): The system temperature [K].
        T0 (float): The reference (environment) temperature [K].
        fluid (optional): If provided, modifies the entropy calculation using a logarithmic relation.
    Returns:
        tuple:
            entropy_term (float): The calculated entropy term.
            exergy_term (float): The calculated exergy term.
    """
    entropy_term = energy_term / Tsys
    
    if fluid:
        if Tsys - T0 != 0:
            entropy_term = energy_term * math.log(Tsys/T0) / (Tsys - T0)
        elif Tsys - T0 == 0:
            entropy_term = 0
            
    exergy_term = energy_term - entropy_term * T0

    if not fluid and Tsys < T0: # Cool exergy (fluid의 경우 항상 exergy term이 양수임 엑서지 항을 구성하는 {(A-B)-ln(A/B)*B} 구조는 항상 A>0, B>0일 때 양수일 수 밖에 없기 때문)
        exergy_term = -exergy_term
    return entropy_term, exergy_term

#%%
# class - Fan & Pump
@dataclass
class Fan:
    def __post_init__(self): 
        # Fan reference: https://www.krugerfan.com/public/uploads/KATCAT006.pdf
        self.fan1 = {
            'flow rate'  : [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], # [m3/s]
            'pressure'   : [140, 136, 137, 147, 163, 178, 182, 190, 198, 181], # [Pa]
            'efficiency' : [0.43, 0.48, 0.52, 0.55, 0.60, 0.65, 0.68, 0.66, 0.63, 0.52], # [-]
            'fan type' : 'centrifugal',
        }
        self.fan2 = {
            'flow rate'  : [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5], # [m3/s]
            'pressure'   : [137, 138, 143, 168, 182, 191, 198, 200, 201, 170], # [Pa]
            'efficiency' : [0.45, 0.49, 0.57, 0.62, 0.67, 0.69, 0.68, 0.67, 0.63, 0.40], # [-]
            'fan type' : 'centrifugal',
        },
        self.fan3 = { # https://ventilatorry.ru/downloads/ebmpapst/datasheet/w3g710-go81-01-en-datasheet-ebmpapst.pdf
            'flow rate' : [0, 6245/cu.h2s, 8330/cu.h2s, 10410/cu.h2s, 12610/cu.h2s], # [m3/s]
            'power' : [0, 100, 238, 465, 827], # [-]
            'fan type' : 'axial',
        }
        self.fan_list = [self.fan1, self.fan2, self.fan3]

    def get_efficiency(self, fan, dV_fan):
        if 'efficiency' not in fan:
            raise ValueError("Selected fan does not have efficiency data.")
        self.efficiency_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['efficiency'])
        eff = cubic_function(dV_fan, *self.efficiency_coeffs)
        return eff
    
    def get_pressure(self, fan, dV_fan):
        if 'pressure' not in fan:
            raise ValueError("Selected fan does not have pressure data.")
        self.pressure_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['pressure'])
        pressure = cubic_function(dV_fan, *self.pressure_coeffs)
        return pressure
    
    def get_power(self, fan, dV_fan):
        if 'efficiency' in fan and 'pressure' in fan:
            eff = self.get_efficiency(fan, dV_fan)
            pressure = self.get_pressure(fan, dV_fan)
            power = pressure * dV_fan / eff
        elif 'power' in fan:
            self.power_coeffs, _ = curve_fit(quartic_function, fan['flow rate'], fan['power'])
            power = quartic_function(dV_fan, *self.power_coeffs)
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 압력(pressure) 및 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        data_pairs = [
            ("pressure", "Pressure [Pa]", "Flow Rate vs Pressure"),
            ("efficiency", "Efficiency [-]", "Flow Rate vs Efficiency"),
        ]

        for ax, (key, ylabel, title) in zip(axes, data_pairs):
            print(f"\n{'='*10} {title} {'='*10}")
            for i, fan in enumerate(self.fan_list):
                # 원본 데이터 (dot 형태)
                ax.scatter(fan['flow rate'], fan[key], label=f'Fan {i+1} Data', color=scatter_colors[i], s=2)

                # 곡선 피팅 수행
                coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan[key])
                flow_range = np.linspace(min(fan['flow rate']), max(fan['flow rate']), 100)
                fitted_values = cubic_function(flow_range, *coeffs)

                # 피팅된 곡선 (line 형태)
                ax.plot(flow_range, fitted_values, label=f'Fan {i+1} Fit', color=plot_colors[i], linestyle='-')
                a,b,c,d = coeffs
                print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

            ax.set_xlabel('Flow Rate [m$^3$/s]', fontsize=dm.fs(0.5))
            ax.set_ylabel(ylabel, fontsize=dm.fs(0.5))
            ax.set_title(title, fontsize=dm.fs(0.5))
            ax.legend()

        plt.subplots_adjust(wspace=0.3)
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

    def show_graph(self):
        """
        유량(flow rate) 대비 압력(pressure) 및 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        data_pairs = [
            ("pressure", "Pressure [Pa]", "Flow Rate vs Pressure"),
            ("efficiency", "Efficiency [-]", "Flow Rate vs Efficiency"),
        ]

        for ax, (key, ylabel, title) in zip(axes, data_pairs):
            print(f"\n{'='*10} {title} {'='*10}")
            for i, fan in enumerate(self.fan_list):
                # 원본 데이터 (dot 형태)
                ax.scatter(fan['flow rate'], fan[key], label=f'Fan {i+1} Data', color=scatter_colors[i], s=2)

                # 곡선 피팅 수행
                coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan[key])
                flow_range = np.linspace(min(fan['flow rate']), max(fan['flow rate']), 100)
                fitted_values = cubic_function(flow_range, *coeffs)

                # 피팅된 곡선 (line 형태)
                ax.plot(flow_range, fitted_values, label=f'Fan {i+1} Fit', color=plot_colors[i], linestyle='-')
                a,b,c,d = coeffs
                print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

            ax.set_xlabel('Flow Rate [m$^3$/s]', fontsize=dm.fs(0.5))
            ax.set_ylabel(ylabel, fontsize=dm.fs(0.5))
            ax.set_title(title, fontsize=dm.fs(0.5))
            ax.legend()

        plt.subplots_adjust(wspace=0.3)
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

@dataclass
class Pump:
    """
    Pump 클래스: 펌프의 성능 데이터를 저장하고 분석하는 클래스.
    
    - 유량(flow rate)과 효율(efficiency) 데이터를 보유.
    - 효율 데이터를 기반으로 곡선 피팅(curve fitting)을 수행하여 예측 값 계산.
    - 주어진 압력 차이(dP_pmp)와 유량(V_pmp)을 이용하여 펌프의 전력 사용량 계산.
    """

    def __post_init__(self):
        """
        클래스 초기화 후 자동 실행되는 메서드.
        두 개의 펌프의 유량 및 효율 데이터를 저장.
        """
        self.pump1 = {
            'flow rate'  : np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])/cu.h2s, # m3/s
            'efficiency' : [0.255, 0.27, 0.3, 0.33, 0.34, 0.33, 0.32, 0.3, 0.26], # [-]
        }
        self.pump2 = {
            'flow rate'  : np.array([1.8, 2.2, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8])/cu.h2s, # m3/s
            'efficiency' : [0.23, 0.26, 0.29, 0.32, 0.35, 0.34, 0.33, 0.31, 0.28], # [-]
        }
        self.pump_list = [self.pump1, self.pump2]
        
    def get_efficiency(self, pump, dV_pmp):
        """
        주어진 유량(V_pmp)에 대해 3차 곡선 피팅을 통해 펌프 효율을 예측.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :return: 예측된 펌프 효율
        """
        self.efficiency_coeffs, _ = curve_fit(cubic_function, pump['flow rate'], pump['efficiency'])
        eff = cubic_function(dV_pmp, *self.efficiency_coeffs)
        return eff

    def get_power(self, pump, V_pmp, dP_pmp):
        """
        주어진 유량(V_pmp)과 압력 차이(dP_pmp)를 이용하여 펌프의 전력 사용량을 계산.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :param dP_pmp: 펌프 압력 차이 (Pa)
        :return: 펌프의 사용 전력 (W)
        """
        efficiency = self.get_efficiency(pump, V_pmp)
        power = (V_pmp * dP_pmp) / efficiency
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        for i, pump in enumerate(self.pump_list):
            # 원본 데이터 (dot 형태)
            ax.scatter(pump['flow rate']*cu.h2s, pump['efficiency'], label=f'Pump {i+1} Data', color=scatter_colors[i], s=2)

            # 곡선 피팅 수행
            coeffs, _ = curve_fit(cubic_function, pump['flow rate']*cu.h2s, pump['efficiency'])
            flow_range = np.linspace(min(pump['flow rate']), max(pump['flow rate']), 100)*cu.h2s
            fitted_values = cubic_function(flow_range, *coeffs)

            # 피팅된 곡선 (line 형태)
            a,b,c,d = coeffs
            ax.plot(flow_range, fitted_values, label=f'Pump {i+1} Fit', color=plot_colors[i], linestyle='-')
            print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

        ax.set_xlabel('Flow Rate [m$^3$/h]', fontsize=dm.fs(0.5))
        ax.set_ylabel('Efficiency [-]', fontsize=dm.fs(0.5))
        ax.legend()

        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

#%%
@dataclass
class HeatPumpModel:
    """
    물리적 원리에 기반한 히트펌프 성능 계산 및 최적 운전점 탐색 클래스.
    """
    def __init__(self,
                 refrigerant='R410A',
                 compressor_displacement_m3=0.0005,
                 eta_compressor_isentropic=0.7,
                 eta_compressor_volumetric=0.85,
                 fan_power_per_airflow_W_per_m3s=500,
                 condenser_approach_temp_K=5.0,
                 evaporator_approach_temp_K=5.0,
                 T_ia_C=20.0):
        """
        히트펌프의 고정된 물리적 파라미터를 초기화합니다.

        Args:
            refrigerant (str): 사용할 냉매 이름 (CoolProp 형식).
            compressor_displacement_m3 (float): 압축기 행정 체적 (1회전 당 흡입량) [m^3].
            eta_compressor_isentropic (float): 압축기 단열 효율. - 단열 효율은 압축 과정에서 발생하는 에너지 손실이 얼마나 적은가를 나타내는 지표
            eta_compressor_volumetric (float): 압축기 체적 효율. - 압축기가 한 번 회전할 때 이론적으로 빨아들일 수 있는 냉매량 대비, 실제로 얼마나 빨아들였는가를 나타내는 지표
            fan_power_per_airflow_W_per_m3s (float): 풍량 대비 팬 동력 계수 [W / (m^3/s)].
            condenser_approach_temp_K (float): 응축기 접근 온도차 (응축온도 - 실내온도) [K].
            evaporator_approach_temp_K (float): 증발기 접근 온도차 (실외온도 - 증발온도) [K].
            T_ia_C (float): 목표 실내 공기 온도 [°C].
        """
        
        self.refrigerant = refrigerant
        self.compressor_displacement_m3 = compressor_displacement_m3
        self.eta_comp_isen = eta_compressor_isentropic
        self.eta_comp_vol = eta_compressor_volumetric
        self.fan_power_coeff = fan_power_per_airflow_W_per_m3s # -> 팬 데이터를 참고하여 설정
        self.cond_approach_K = condenser_approach_temp_K
        self.evap_approach_K = evaporator_approach_temp_K
        self.T_ia_C = T_ia_C
        self.T_ia_K = cu.C2K(self.T_ia_C)

    def _calculate_cycle_performance(self, comp_speed_rps, fan_airflow_m3s, T0_C):
        """
        주어진 운전 조건(압축기/팬 속도, 외기온도)에서 사이클 성능을 계산하는 내부 함수.
        (저온/저압 가스)                                (고온/고압 가스)
        (1) -------------------- [ 압축기 ] --------------------> (2)
        ^                                                        |
        |                                                        v
        |                                                        |
        [증발기]                                                [응축기]
        [실외기]                                                [실내기]
        (열 흡수 ❄️)                                           (열 방출 🔥)
        ^                                                        |
        |                                                        v
        |                                                        |
        (4) <----------------- [ 팽창밸브 ] <------------------- (3)
        (저온/저압 액체+가스)                                     (고압 액체)
        """
        
        # --- 1. 증발 및 응축 온도/압력 계산 ---
        # 증발 온도 = 외기온도 - 접근온도차
        T_evap_C = T0_C - self.evap_approach_K
        P_evap_Pa = CP.PropsSI('P', 'T', cu.C2K(T_evap_C), 'Q', 1, self.refrigerant)

        # 응축 온도 = 실내온도 + 접근온도차
        T_cond_C = self.T_ia_C + self.cond_approach_K
        P_cond_Pa = CP.PropsSI('P', 'T', cu.C2K(T_cond_C), 'Q', 0, self.refrigerant)

        # --- 2. 사이클의 각 지점(State 1, 2, 3, 4) 물성치 계산 ---
        # State 1: 압축기 입구 (포화 증기)
        h1 = CP.PropsSI('H', 'P', P_evap_Pa, 'Q', 1, self.refrigerant)  # J/kg
        s1 = CP.PropsSI('S', 'P', P_evap_Pa, 'Q', 1, self.refrigerant)  # J/kg-K
        rho1 = CP.PropsSI('D', 'P', P_evap_Pa, 'Q', 1, self.refrigerant) # kg/m^3

        # State 2: 압축기 출구 (과열 증기)
        # 등엔트로피 압축 후의 엔탈피(h2s) 계산
        h2s = CP.PropsSI('H', 'P', P_cond_Pa, 'S', s1, self.refrigerant)
        # 실제 압축 후의 엔탈피(h2) 계산 (단열효율 적용)
        h2 = h1 + (h2s - h1) / self.eta_comp_isen

        # State 3: 응축기 출구 (포화 액체)
        h3 = CP.PropsSI('H', 'P', P_cond_Pa, 'Q', 0, self.refrigerant)

        # State 4: 팽창밸브 출구 (이상 팽창, 등엔탈피 과정)
        h4 = h3

        # --- 3. 질량 유량 및 성능 지표 계산 ---
        # 질량 유량 (m_dot) = 회전수(회전수/s) * 행정체적(1회전당 흡입량 m3/회전수) * 흡입밀도(kg/m3) * 체적효율(실제 흡입량/이론 흡입량)
        m_dot = comp_speed_rps * self.compressor_displacement_m3 * rho1 * self.eta_comp_vol

        # 난방 능력 (응축기 방출 열량)
        heating_capacity_kW = m_dot * (h2 - h3) / 1000.0

        # 압축기 사용 전력
        compressor_power_kW = m_dot * (h2 - h1) / 1000.0

        ##########################################################################
        # 팬 사용 전력
        fan_power_kW = (fan_airflow_m3s * self.fan_power_coeff) / 1000.0
        ##########################################################################
        
        total_power_kW = compressor_power_kW + fan_power_kW
        
        # COP (Coefficient of Performance -> system energy efficiency)
        cop = heating_capacity_kW / total_power_kW if total_power_kW > 0 else 0

        return {
            "heating_capacity_kW": heating_capacity_kW,
            "compressor_power_kW": compressor_power_kW,
            "fan_power_kW": fan_power_kW,
            "total_power_kW": total_power_kW,
            "cop": cop,
            "m_dot_kg_s": m_dot,
            "T_evap_C": T_evap_C,
            "P_evap_kPa": P_evap_Pa / 1000.0,
            "T_cond_C": T_cond_C,
            "P_cond_kPa": P_cond_Pa / 1000.0,
        }

    def find_optimal_operation(self, required_heating_load_kW, T0_C):
        """
        주어진 난방 부하와 외기온도 조건에서 총 전력사용를 최소화하는
        압축기 및 팬 운전 조건을 찾습니다.

        Args:
            required_heating_load_kW (float): 요구되는 난방 부하 [kW].
            T0_C (float): 실외 공기 온도 [°C].

        Returns:
            dict: 최적화 결과 또는 에러 메시지.
        """
        # 최적화 변수: x[0] = 압축기 회전수(rps), x[1] = 팬 풍량(m^3/s)
        
        # 1. 목적 함수: 총 전력 사용량 (최소화 대상)
        def objective(x):
            comp_speed, fan_airflow = x
            perf = self._calculate_cycle_performance(comp_speed, fan_airflow, T0_C)
            return perf["total_power_kW"]

        # 2. 제약 조건: 계산된 난방 능력이 요구 부하와 같아야 함
        def constraint(x):
            comp_speed, fan_airflow = x
            perf = self._calculate_cycle_performance(comp_speed, fan_airflow, T0_C)
            # solver가 0을 만족하는 해를 찾으므로 (계산값 - 목표값) 형태로 반환
            return perf["heating_capacity_kW"] - required_heating_load_kW

        # 변수의 경계 조건 (최소/최대 운전 범위)
        # 압축기: 10 ~ 100 rps (600 ~ 6000 rpm), 팬: 0.1 ~ 3.0 m^3/s
        bounds = [(10, 100), (0.1, 3.0)]
        
        # 제약 조건 설정
        cons = ({'type': 'eq', 'fun': constraint})
        
        # 초기 추정값
        initial_guess = [40, 0.8]

        # 최적화 실행 (SLSQP 알고리즘 사용)
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=cons, options={'disp': False})

        if result.success:
            optimal_comp_speed, optimal_fan_airflow = result.x
            final_performance = self._calculate_cycle_performance(
                optimal_comp_speed, optimal_fan_airflow, T0_C
            )
            
            # 보기 쉽게 결과 정리
            output = {
                "success": True,
                "message": "최적 운전점을 찾았습니다.",
                "required_load_kW": required_heating_load_kW,
                "T0_C": T0_C,
                "optimal_compressor_speed_rps": round(optimal_comp_speed, 2),
                "optimal_compressor_speed_rpm": round(optimal_comp_speed * 60, 0),
                "optimal_fan_airflow_m3s": round(optimal_fan_airflow, 3),
                "performance": {
                    "Calculated_Heating_Capacity_kW": round(final_performance["heating_capacity_kW"], 3),
                    "COP": round(final_performance["cop"], 3),
                    "Total_Power_kW": round(final_performance["total_power_kW"], 3),
                    "Compressor_Power_kW": round(final_performance["compressor_power_kW"], 3),
                    "Fan_Power_kW": round(final_performance["fan_power_kW"], 3),
                    "Evaporating_Temp_C": round(final_performance["T_evap_C"], 2),
                    "Condensing_Temp_C": round(final_performance["T_cond_C"], 2),
                }
            }
            return output
        else:
            return {
                "success": False,
                "message": f"최적화에 실패했습니다: {result.message}"
            }


def plot_cycle_diagrams(refrigerant, states):
    """
    계산된 사이클 상태(1,2,3,4)를 바탕으로 p-h, T-h 선도를 그립니다.
    """
    # colors
    color1 = 'dm.blue5'
    color2 = 'dm.red5'
    color3 = 'dm.black'

    ymin1, ymax1, yint1 = 0, 10**4, 0
    ymin2, ymax2, yint2 = -40, 80, 20
    xmin, xmax, xint = 0, 500, 100

    # --- 임계/포화 데이터 준비 ---
    # (CoolProp 순서는 PropsSI('키', 유체명) 입니다)
    T_critical = cu.K2C(CP.PropsSI('Tcrit',  refrigerant))
    P_critical = CP.PropsSI('Pcrit',  refrigerant) / 1000  # kPa (참고용, 여기선 미사용)

    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 200)
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]
    p_sat = [CP.PropsSI('P', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]

    # 상태값(kPa, kJ/kg, °C)
    p = [states[i]['P'] for i in range(1, 5)]
    h = [states[i]['H'] for i in range(1, 5)]
    T = [states[i]['T'] for i in range(1, 5)]

    # 사이클 경로(닫기)
    h_cycle = h + [h[0]]
    p_cycle = p + [p[0]]
    T_cycle = T + [T[0]]

    # --- Figure & Axes ---
    LW = np.arange(0.5, 3.0, 0.25)
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(figsize=(dm.cm2in(20), dm.cm2in(7)), nrows=nrows, ncols=ncols)
    ax = np.atleast_1d(ax).ravel()  # 1D 인덱싱

    # 축별 메타데이터(인덱스로 접근)
    xlabels = ['Enthalpy [kJ/kg]', 'Enthalpy [kJ/kg]']
    ylabels = ['Pressure (log scale) [kPa]', 'Temperature [°C]']
    yscales = ['log', 'linear']
    xlims   = [(xmin, xmax), (xmin, xmax)]
    ylims   = [(ymin1, ymax1), (ymin2, ymax2)]

    # 포화선/사이클 Y데이터 선택자
    satY_list   = [p_sat, temps]          # idx=0: p_sat vs h, idx=1: T(temps) vs h
    cycleY_list = [p_cycle, T_cycle]

    # 상태 텍스트 Y좌표 함수(축별로 다르게)
    def state_y(idx, i):
        return p[i]*1.1 if idx == 0 else (T[i] + yint2*0.1)

    # 공통 범례 스타일
    legend_kw = dict(
        loc='upper left',
        bbox_to_anchor=(0.0, 0.99),
        handlelength=1.5,
        labelspacing=0.5,
        columnspacing=2,
        ncol=1,
        frameon=False,
        fontsize=dm.fs(-1)
    )

    # --- 2중 for문으로 그리기 ---
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            axi = ax[idx]

            # 포화선
            axi.plot(h_liq, satY_list[idx],  color=color1, label='Saturated Liquid', linewidth=LW[2])
            axi.plot(h_vap, satY_list[idx],  color=color2, label='Saturated Vapor',  linewidth=LW[2])
            # 사이클 경로
            axi.plot(h_cycle, cycleY_list[idx], color=color3, label='Heat Pump Cycle',
                     linewidth=LW[1], marker='o', linestyle=':', markersize=2)

            # 상태 라벨
            for i in range(4):
                axi.text(h[i]*1.01, state_y(idx, i), f'State {i+1}',
                         fontsize=dm.fs(-1), ha='center', va='bottom')

            # 축 설정
            axi.set_xlabel(xlabels[idx], fontsize=dm.fs(0))
            axi.set_ylabel(ylabels[idx], fontsize=dm.fs(0))
            axi.set_yscale(yscales[idx])
            axi.set_xlim(*xlims[idx])
            axi.set_ylim(*ylims[idx])
            axi.legend(**legend_kw)

    dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
    plt.savefig('../../figure/HeatPump_model/HeatPump_Cycle_Diagram.png', dpi=600)
    dm.save_and_show(fig)
    

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # 1. 히트펌프 모델 객체 생성
    my_heat_pump = HeatPumpModel(
        refrigerant='R410A',
        compressor_displacement_m3=0.000045,
        eta_compressor_isentropic=0.75,
        eta_compressor_volumetric=0.9,
        fan_power_per_airflow_W_per_m3s=500,
        condenser_approach_temp_K=5.0,
        evaporator_approach_temp_K=5.0,
        T_ia_C=20.0
    )

    # 2. 시뮬레이션 조건 설정
    load_condition = {"required_heating_load_kW": 5.0, "T0_C": 2.0}
    
    # 3. 최적 운전점 탐색
    optimal_result = my_heat_pump.find_optimal_operation(**load_condition)

    if optimal_result["success"]:
        print("--- 최적 운전 결과 ---")
        # ... (이전과 동일한 결과 출력 부분) ...
        print(f"COP: {optimal_result['performance']['COP']}")
        print(f"난방 능력 (kW): {optimal_result['performance']['Calculated_Heating_Capacity_kW']}")
        print(f"총 전력 사용량 (kW): {optimal_result['performance']['Total_Power_kW']}")
        print(f"압축기 전력 사용량 (kW): {optimal_result['performance']['Compressor_Power_kW']}")
        print(f" 팬 전력 사용량 (kW): {optimal_result['performance']['Fan_Power_kW']}")
        print(f"증발 온도 (°C): {optimal_result['performance']['Evaporating_Temp_C']}")
        print(f"응축 온도 (°C): {optimal_result['performance']['Condensing_Temp_C']}")
        print(f"팬 풍량 (m3/s): {optimal_result['optimal_fan_airflow_m3s']}")
        print(f"압축기 회전수 (RPM): {optimal_result['optimal_compressor_speed_rpm']}")

        # 4. 그래프 그리기를 위한 상태값 계산 및 저장
        # 최적화된 속도로 다시 한번 사이클 계산을 수행하여 각 지점의 물성치 확보
        opt_speed = optimal_result['optimal_compressor_speed_rps']
        opt_airflow = optimal_result['optimal_fan_airflow_m3s']

        # 각 상태의 압력(P), 엔탈피(H), 온도(T)를 계산
        P_evap = CP.PropsSI('P', 'T', cu.C2K(optimal_result['performance']['Evaporating_Temp_C']), 'Q', 1, my_heat_pump.refrigerant)
        P_cond = CP.PropsSI('P', 'T', cu.C2K(optimal_result['performance']['Condensing_Temp_C']), 'Q', 0, my_heat_pump.refrigerant)

        states_data = {}
        # State 1
        states_data[1] = {
            'P': P_evap / 1000, # Pa -> kPa
            'H': CP.PropsSI('H', 'P', P_evap, 'Q', 1, my_heat_pump.refrigerant) / 1000, # J/kg -> kJ/kg
            'T': cu.K2C(CP.PropsSI('T', 'P', P_evap, 'Q', 1, my_heat_pump.refrigerant))
        }
        # State 2
        s1 = CP.PropsSI('S', 'P', P_evap, 'Q', 1, my_heat_pump.refrigerant)
        h2s = CP.PropsSI('H', 'P', P_cond, 'S', s1, my_heat_pump.refrigerant)
        h2 = (states_data[1]['H']*1000 + (h2s - states_data[1]['H']*1000) / my_heat_pump.eta_comp_isen)
        states_data[2] = {
            'P': P_cond / 1000,
            'H': h2 / 1000,
            'T': cu.K2C(CP.PropsSI('T', 'P', P_cond, 'H', h2, my_heat_pump.refrigerant))
        }
        # State 3
        states_data[3] = {
            'P': P_cond / 1000,
            'H': CP.PropsSI('H', 'P', P_cond, 'Q', 0, my_heat_pump.refrigerant) / 1000, # J/kg -> kJ/kg
            'T': cu.K2C(CP.PropsSI('T', 'P', P_cond, 'Q', 0, my_heat_pump.refrigerant))
        }
        # State 4
        states_data[4] = {
            'P': P_evap / 1000, # Pa -> kPa     
            'H': states_data[3]['H'], # h4 = h3
            'T': cu.K2C(CP.PropsSI('T', 'P', P_evap, 'H', states_data[3]['H']*1000, my_heat_pump.refrigerant))
        }

        # 5. 그래프 그리기 함수 호출
        plot_cycle_diagrams(my_heat_pump.refrigerant, states_data)
        
    else:
        print(f"계산 실패: {optimal_result['message']}")
 
# %%
