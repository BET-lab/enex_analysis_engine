import numpy as np
import math
from . import calc_util as cu
# import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import erf

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

def calculate_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_r_int, Q_r_max, COP_ref):
    '''
    https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in cooling mode.

    Parameters:
    - T_a_int_out : Indoor air temperature [K]
    - T_a_ext_in  : Outdoor air temperature [K]
    - Q_r_int     : Indoor heat load [W]
    - Q_r_max     : Maximum cooling capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    - EIR : Energy input to cooling output ratio
    - COP_ref : the reference COP at the standard conditions
    '''
    PLR = Q_r_int / Q_r_max
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP

def calculate_ASHP_heating_COP(T0, Q_r_int, Q_r_max):
    '''
    https://www.mdpi.com/2071-1050/15/3/1880
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in heating mode.

    Parameters:
    - T0 : Enviromnetal temperature [K]
    - Q_r_int : Indoor heat load [W]
    - Q_r_max : Maximum heating capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    '''
    PLR = Q_r_int / Q_r_max
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
        entropy_term = energy_term *math.log(Tsys/T0)/(Tsys - T0)
    exergy_term = energy_term - T0 * entropy_term
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
        }
        self.fan2 = {
            'flow rate'  : [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5], # [m3/s]
            'pressure'   : [137, 138, 143, 168, 182, 191, 198, 200, 201, 170], # [Pa]
            'efficiency' : [0.45, 0.49, 0.57, 0.62, 0.67, 0.69, 0.68, 0.67, 0.63, 0.40], # [-]
        }
        self.fan_list = [self.fan1, self.fan2]

    def get_effieciency(self, fan, dV_fan):
        self.efficiency_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['efficiency'])
        eff = cubic_function(dV_fan, *self.efficiency_coeffs)
        return eff
    
    def get_pressure(self, fan, dV_fan):
        self.pressure_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['pressure'])
        pressure = cubic_function(dV_fan, *self.pressure_coeffs)
        return pressure
    
    def get_power(self, fan, dV_fan):
        eff = self.get_effieciency(fan, dV_fan)
        pressure = self.get_pressure(fan, dV_fan)
        power = pressure * dV_fan / eff
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

@dataclass
class Pump:
    """
    Pump 클래스: 펌프의 성능 데이터를 저장하고 분석하는 클래스.
    
    - 유량(flow rate)과 효율(efficiency) 데이터를 보유.
    - 효율 데이터를 기반으로 곡선 피팅(curve fitting)을 수행하여 예측 값 계산.
    - 주어진 압력 차이(dP_pmp)와 유량(V_pmp)을 이용하여 펌프의 전력 소비량 계산.
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
        주어진 유량(V_pmp)과 압력 차이(dP_pmp)를 이용하여 펌프의 전력 소비량을 계산.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :param dP_pmp: 펌프 압력 차이 (Pa)
        :return: 펌프의 소비 전력 (W)
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
# class - Domestic Hot Water System
@dataclass
class ElectricBoiler:

    def __post_init__(self):
        subsystem_category = ['hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Temperature [K]
        self.T_w_tank = 60
        self.T_w_sup  = 10
        self.T_w_serv = 45
        self.T0       = 0

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T_w_tank = cu.C2K(self.T_w_tank) # tank water temperature [K]
        self.T_w_sup  = cu.C2K(self.T_w_sup)  # supply water temperature [K]
        self.T_w_serv  = cu.C2K(self.T_w_serv)  # tap water temperature [K]
        self.T0       = cu.C2K(self.T0)       # reference temperature [K]
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_tank = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot

        # Heat Transfer Rates
        self.Q_w_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)
        self.E_heater = self.Q_w_tank + self.Q_l_tank - self.Q_w_sup # Electric Power input [W]

        # Pre-calculate Energy values
        self.Q_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_heater, self.X_heater = generate_entropy_exergy_term(self.E_heater, float('inf'), self.T0)
        self.S_w_sup_tank, self.X_w_sup_tank = generate_entropy_exergy_term(self.Q_w_sup_tank, self.T_w_sup, self.T0, fluid=True)
        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)

        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - (self.S_heater + self.S_w_sup_tank)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        # Pre-calculate Exergy values for hot water tank
        self.X_c_tank = self.S_g_tank * self.T0

        # Pre-calculate Exergy values for mixing valve
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_heater

        # Energy Balance ========================================
        # hot water tank energy balance (without using lists)
        self.energy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.energy_balance["hot water tank"]["in"]["Q_w_sup_tank"] = self.Q_w_sup_tank
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        # Mixing valve energy balance (without using lists)
        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["hot water tank"]["in"]["S_heater"] = self.S_heater
        self.entropy_balance["hot water tank"]["in"]["S_w_sup_tank"] = self.S_w_sup_tank
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class GasBoiler:

    def __post_init__(self):
        subsystem_category = ['combustion chamber', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_comb = 0.9

        # Temperature [°C]
        self.T_w_tank = 60 
        self.T_w_sup  = 10
        self.T_w_serv  = 45 
        self.T0       = 0
        self.T_exh    = 70 

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T_w_tank = cu.C2K(self.T_w_tank) # tank water temperature [K]
        self.T_w_sup  = cu.C2K(self.T_w_sup)  # supply water temperature [K]
        self.T_w_serv  = cu.C2K(self.T_w_serv)  # tap water temperature [K]
        self.T0       = cu.C2K(self.T0)       # reference temperature [K]
        self.T_exh    = cu.C2K(self.T_exh)    # exhaust gas temperature [K]

        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        self.exergy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.exergy_balance["combustion chamber"]["in"]["X_w_sup"] = self.X_w_sup
        self.exergy_balance["combustion chamber"]["out"]["X_w_comb_out"] = self.X_w_comb_out
        self.exergy_balance["combustion chamber"]["out"]["X_exh"] = self.X_exh
        self.exergy_balance["combustion chamber"]["con"]["X_c_comb"] = self.X_c_comb

        self.exergy_balance["hot water tank"]["in"]["X_w_comb_out"] = self.X_w_comb_out
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        subsystem_category = ['external unit', 'refrigerant', 'hot water tank', 'mixing valve']
        self.energy_balance, self.entropy_balance, self.exergy_balance = generate_balance_dict(subsystem_category)
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.eta_comb = 0.9

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 

        # Maximum heat transfer from refrigerant to tank water [W]
        self.Q_r_max = 4000

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv    = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy and Exergy values
        self.S_NG, self.X_NG_term = generate_entropy_exergy_term(self.E_NG, self.T_NG, self.T0)
        self.X_NG = ex_eff_NG * self.E_NG
        self.S_w_sup, self.X_w_sup = generate_entropy_exergy_term(self.Q_w_sup, self.T_w_sup, self.T0, fluid=True)
        self.S_w_comb_out, self.X_w_comb_out = generate_entropy_exergy_term(self.Q_w_comb_out, self.T_w_comb, self.T0, fluid=True)
        self.S_exh, self.X_exh = generate_entropy_exergy_term(self.Q_exh, self.T_exh, self.T0)
        
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)
        self.X_c_comb = self.S_g_comb * self.T0

        self.S_w_tank, self.X_w_tank = generate_entropy_exergy_term(self.Q_w_tank, self.T_w_tank, self.T0, fluid=True)
        self.S_l_tank, self.X_l_tank = generate_entropy_exergy_term(self.Q_l_tank, self.T_tank_is, self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out
        self.X_c_tank = self.S_g_tank * self.T0

        self.S_w_sup_mix, self.X_w_sup_mix = generate_entropy_exergy_term(self.Q_w_sup_mix, self.T_w_sup, self.T0, fluid=True)
        self.S_w_serv, self.X_w_serv = generate_entropy_exergy_term(self.Q_w_serv, self.T_w_serv, self.T0, fluid=True)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance["combustion chamber"]["in"]["E_NG"] = self.E_NG
        self.energy_balance["combustion chamber"]["in"]["Q_w_sup"] = self.Q_w_sup
        self.energy_balance["combustion chamber"]["out"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["combustion chamber"]["out"]["Q_exh"] = self.Q_exh

        self.energy_balance["hot water tank"]["in"]["Q_w_comb_out"] = self.Q_w_comb_out
        self.energy_balance["hot water tank"]["out"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["hot water tank"]["out"]["Q_l_tank"] = self.Q_l_tank

        self.energy_balance["mixing valve"]["in"]["Q_w_tank"] = self.Q_w_tank
        self.energy_balance["mixing valve"]["in"]["Q_w_sup_mix"] = self.Q_w_sup_mix
        self.energy_balance["mixing valve"]["out"]["Q_w_serv"] = self.Q_w_serv

        ## Entropy Balance ========================================
        self.entropy_balance["combustion chamber"]["in"]["S_NG"] = self.S_NG
        self.entropy_balance["combustion chamber"]["in"]["S_w_sup"] = self.S_w_sup
        self.entropy_balance["combustion chamber"]["out"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["combustion chamber"]["out"]["S_exh"] = self.S_exh
        self.entropy_balance["combustion chamber"]["gen"]["S_g_comb"] = self.S_g_comb

        self.entropy_balance["hot water tank"]["in"]["S_w_comb_out"] = self.S_w_comb_out
        self.entropy_balance["hot water tank"]["out"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["hot water tank"]["out"]["S_l_tank"] = self.S_l_tank
        self.entropy_balance["hot water tank"]["gen"]["S_g_tank"] = self.S_g_tank
        
        self.entropy_balance["mixing valve"]["in"]["S_w_tank"] = self.S_w_tank
        self.entropy_balance["mixing valve"]["in"]["S_w_sup_mix"] = self.S_w_sup_mix
        self.entropy_balance["mixing valve"]["out"]["S_w_serv"] = self.S_w_serv
        self.entropy_balance["mixing valve"]["gen"]["S_g_mix"] = self.S_g_mix

        ## Exergy Balance ========================================
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"]["in"]["E_heater"] = self.E_heater
        self.exergy_balance["hot water tank"]["in"]["X_w_sup_tank"] = self.X_w_sup_tank
        self.exergy_balance["hot water tank"]["out"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["hot water tank"]["out"]["X_l_tank"] = self.X_l_tank
        self.exergy_balance["hot water tank"]["con"]["X_c_tank"] = self.X_c_tank

        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"]["in"]["X_w_tank"] = self.X_w_tank
        self.exergy_balance["mixing valve"]["in"]["X_w_sup_mix"] = self.X_w_sup_mix
        self.exergy_balance["mixing valve"]["out"]["X_w_serv"] = self.X_w_serv
        self.exergy_balance["mixing valve"]["con"]["X_c_mix"] = self.X_c_mix

 #%% 
# class - Electric heater
@dataclass
class ElectricHeater:

    def __post_init__(self): 
        
        # hb: heater body
        # hs: heater surface
        # ms: room surface
        
        # Heater material properties (냉간압연 탄소강판 SPCC)
        self.c   = 500 # [J/kgK]
        self.rho = 7800 # [kg/m3]
        self.k   = 50 # [W/mK]
    
        # Heater geometry [m]
        self.D = 0.005 
        self.H = 0.8 
        self.W = 1.0
        
        # Electricity input to the heater [W]
        self.E_heater = 1000
        
        # Temperature [°C]
        self.T0   = 0
        self.T_mr = 15
        self.T_init = 20 # Initial temperature of the heater [°C]
        self.T_a_room = 20 # Indoor air temperature [°C]
        
        # Emissivity [-]
        self.epsilon_hs = 1 # hs: heater surface
        self.epsilon_rs = 1 # rs: room surface
        
        # Time step [s]
        self.dt = 10
    
    def system_update(self):
        
        # Temperature [K]
        self.T0     = cu.C2K(self.T0) # 두번 system update를 할 경우 절대온도 변환 중첩됨
        self.T_mr   = cu.C2K(self.T_mr)
        self.T_a_room   = cu.C2K(self.T_a_room)
        self.T_init = cu.C2K(self.T_init)
        self.T_hb   = self.T_init # hb: heater body
        self.T_hs   = self.T_init # hs: heater surface
        
        # Heater material properties
        self.C = self.c * self.rho
        self.A = self.H * self.W * 2 # double side 
        self.V = self.H * self.W * self.D
        
        # Conductance [W/m²K]
        self.K_cond = self.k / (self.D / 2)
        
        # Iterative calculation
        self.time = []
        self.T_hb_list = []
        self.T_hs_list = []
        
        self.E_heater_list = []
        self.Q_st_list = []
        self.Q_cond_list = []
        self.Q_conv_list = []
        self.Q_rad_hs_list = []
        self.Q_rad_rs_list = []
        
        self.S_st_list = []
        self.S_heater_list = []
        self.S_cond_list = []
        self.S_conv_list = []
        self.S_rad_rs_list = []
        self.S_rad_hs_list = []
        self.S_g_hb_list = []
        self.S_g_hs_list = []
        
        self.X_st_list = [] 
        self.X_heater_list = []
        self.X_cond_list = []
        self.X_conv_list = []
        self.X_rad_rs_list = []
        self.X_rad_hs_list = []
        self.X_c_hb_list = []
        self.X_c_hs_list = []
        
        index = 0
        tolerance = 1e-8
        while True:
            self.time.append(index * self.dt)
            
            # Heat transfer coefficient [W/m²K]
            self.h_cp = calc_h_vertical_plate(self.T_hs, self.T0, self.H) 
            
            def residual_Tp(Tp_new):
                # 축열 항
                Q_st = self.rho * self.c * self.V * (Tp_new - self.T_hb) / self.dt

                # Tps 계산 (표면에너지 평형으로부터)
                Tps = (
                    self.K_cond * Tp_new
                    + self.h_cp * self.T_a_room
                    + self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr**4 - self.T0**4)
                    - self.epsilon_hs * self.epsilon_rs * sigma * (Tp_new**4 - self.T0**4)
                ) / (self.K_cond + self.h_cp)

                # 전도열
                Q_cond = self.A * self.K_cond * (Tp_new - Tps)

                return Q_st + Q_cond - self.E_heater
            
            self.T_hb_guess = self.T_hb # 초기 추정값
            
            from scipy.optimize import fsolve
            self.T_hb_next = fsolve(residual_Tp, self.T_hb_guess)[0]
            self.T_hb_old = self.T_hb
            
            # Temperature update
            self.T_hb = self.T_hb_next
            
            # T_hs update (Energy balance surface: Q_cond + Q_rad_rs = Q_conv + Q_rad_hs)
            self.T_hs = (
                self.K_cond * self.T_hb
                + self.h_cp * self.T_a_room
                + self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 4 - self.T0 ** 4)
                - self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 4 - self.T0 ** 4)
            ) / (self.K_cond + self.h_cp)
            
            # Temperature [K]
            self.T_hb_list.append(self.T_hb)
            self.T_hs_list.append(self.T_hs)
            
            # Conduction [W]
            self.Q_st = self.C * self.V * (self.T_hb_next - self.T_hb_old) / self.dt
            self.Q_cond = self.A * self.K_cond * (self.T_hb - self.T_hs)
            self.Q_conv = self.A * self.h_cp * (self.T_hs - self.T_a_room) # h_cp 추후 변하게
            self.Q_rad_rs = self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 4 - self.T0 ** 4)
            self.Q_rad_hs = self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 4 - self.T0 ** 4)
            
            self.E_heater_list.append(self.E_heater)
            self.Q_st_list.append(self.Q_st)
            self.Q_cond_list.append(self.Q_cond)
            self.Q_conv_list.append(self.Q_conv)
            self.Q_rad_hs_list.append(self.Q_rad_hs)
            self.Q_rad_rs_list.append(self.Q_rad_rs)
            
            # Entropy balance
            self.S_st = (1/self.T_hb) * (self.Q_st)
            self.S_heater = (1/float('inf')) * (self.E_heater)
            self.S_cond = (1/self.T_hb) * (self.Q_cond)
            self.S_conv = (1/self.T_hs) * (self.Q_conv)
            self.S_rad_rs  = 4/3 * self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 3 - self.T0 ** 3)
            self.S_rad_hs  = 4/3 * self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 3 - self.T0 ** 3)
            self.S_g_hb = self.S_st + self.S_conv - self.S_heater     
            self.S_g_hs = self.S_rad_hs + self.S_conv - self.S_cond - self.S_rad_rs

            self.S_st_list.append(self.S_st)
            self.S_heater_list.append(self.S_heater)
            self.S_cond_list.append(self.S_cond)
            self.S_conv_list.append(self.S_conv)
            self.S_rad_rs_list.append(self.S_rad_rs)
            self.S_rad_hs_list.append(self.S_rad_hs)
            self.S_g_hb_list.append(self.S_g_hb)
            self.S_g_hs_list.append(self.S_g_hs)
            
            # Exergy balance
            self.X_st = (1 - self.T0 / self.T_hb) * (self.Q_st)
            self.X_heater = (1 - self.T0 / float('inf')) * (self.E_heater)
            self.X_cond = (1 - self.T0 / self.T_hb) * (self.Q_cond)
            
            ###########################
            # self.X_conv = (1 - self.T0 / self.T_hs) * (self.Q_conv) # h_cp 추후 변하게
            self.X_conv = (1 - self.T0 / ((self.T_hs+self.T0)/2)) * (self.Q_conv) # 임시 변경 사항있으니 주의 필요 -----------------------------
            ############################
            
            self.X_rad_rs = self.Q_rad_rs - self.T0 * self.S_rad_rs
            self.X_rad_hs = self.Q_rad_hs - self.T0 * self.S_rad_hs
            self.X_c_hb = -(self.X_st + self.X_cond - self.X_heater)
            self.X_c_hs = -(self.X_rad_hs + self.X_conv - self.X_cond - self.X_rad_rs)
            
            self.X_st_list.append(self.X_st)
            self.X_heater_list.append(self.X_heater)
            self.X_cond_list.append(self.X_cond)
            self.X_conv_list.append(self.X_conv)
            self.X_rad_rs_list.append(self.X_rad_rs)
            self.X_rad_hs_list.append(self.X_rad_hs)
            self.X_c_hb_list.append(self.X_c_hb)
            self.X_c_hs_list.append(self.X_c_hs)
            
            index += 1
            T_hb_rel_change = abs(self.T_hb_next - self.T_hb_old) / max(abs(self.T_hb_next), 1e-8)
            if T_hb_rel_change < tolerance:
                break
            
            if index > 10000:
                print("time step is too short")
                break
        
        self.X_eff = (self.X_rad_hs + self.X_conv)/ self.X_heater 
        self.energy_balance = {}
        self.energy_balance["heater body"] = {
            "in": {
                "E_heater": self.E_heater,
            },
            "out": {
                "Q_st": self.Q_st,
                "Q_cond": self.Q_cond
            }
        }

        self.energy_balance["heater surface"] = {
            "in": {
                "Q_cond": self.Q_cond,
                "Q_rad_rs": self.Q_rad_rs,
            },
            "out": {
                "Q_conv": self.Q_conv,
                "Q_rad_hs": self.Q_rad_hs
            }
        }
        
        self.entropy_balance = {}
        self.entropy_balance["heater body"] = {
            "in": {
                "S_heater": self.S_heater,
            },
            "gen": {
                "S_g_hb": self.S_g_hb,
            },
            "out": {
                "S_st": self.S_st,
                "S_cond": self.S_cond
            }
        }

        self.entropy_balance["heater surface"] = {
            "in": {
                "S_cond":   self.S_cond,
                "S_rad_rs": self.S_rad_rs,
            },
            "gen": {
                "S_g_hs": self.S_g_hs,
            },
            "out": {
                "S_conv":   self.S_conv,
                "S_rad_hs": self.S_rad_hs
            }
        }
        
        self.exergy_balance = {}
        self.exergy_balance["heater body"] = {
            "in": {
                "X_heater": self.X_heater,
            },
            "con": {
                "X_c_hb": self.X_c_hb,
            },
            "out": {
                "X_st": self.X_st,
                "X_cond": self.X_cond
            }
        }

        self.exergy_balance["heater surface"] = {
            "in": {
                "X_cond":   self.X_cond,
                "X_rad_rs": self.X_rad_rs,
            },
            "con": {
                "X_c_hs": self.X_c_hs,
            },
            "out": {
                "X_conv":   self.X_conv,
                "X_rad_hs": self.X_rad_hs
            }
        }
        