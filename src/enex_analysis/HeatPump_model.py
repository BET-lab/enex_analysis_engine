#%%
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
import CoolProp.CoolProp as CP
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.optimize import root_scalar

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]

sigma = 5.67*10**-8 # Ste_fan-Boltzmann constant [W/m²K⁴]

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
    
    balance_type = 'energy'
    unit = '[W]'
    
    for subsystem, category_dict in balance.items(): 
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            if 'gen' in category:
                balance_type = 'entropy'
                unit = '[W/K]'
            elif 'con' in category:
                balance_type = 'exergy'
    
    for subsystem, category_dict in balance.items(): 
        # subsystem: hot water tank, mixing valve...
        # category_dict: {in: {a,b}, out: {a,b}...} 
        text = f'{subsystem.upper()} {balance_type.upper()} BALANCE:'
        print(f'\n\n{text}'+"="*(total_length-len(text)))
        
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            # terms: {a,b}
            # a,b..: symbol: value
            print(f'\n{category.upper()} ENTRIES:')
            
            for symbol, value in terms.items():
                print(f'{symbol}: {round(value, decimal)} {unit}')

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

#%%
def calc_simple_tank_UA(
        # Tank size [m]
        r0 = 0.2,
        H = 0.8,
        # Tank layer thickness [m]
        x_shell = 0.01,
        x_ins   = 0.10,
        # Tank thermal conductivity [W/mK]
        k_shell = 25,  
        k_ins   = 0.03,
        # External convective heat transfer coefficient [W/m²K]
        h_o     = 10,
        ):
    
        r1 = r0 + x_shell
        r2 = r1 + x_ins
        
        # Tank surface areas [m²]
        A_side = 2 * math.pi * r2 * H
        A_base = math.pi * r0**2
        R_base_unit = x_shell / k_shell + x_ins / k_ins # [m2K/W]
        R_side_unit = math.log(r1 / r0) / (2 * math.pi * k_shell) + math.log(r2 / r1) / (2 * math.pi * k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        R_base = R_base_unit / A_base # [K/W]
        R_side = R_side_unit / H # [K/W]
        
        # Thermal resistances [K/W]
        R_base_ext = 1 / (h_o * A_base)
        R_side_ext = 1 / (h_o * A_side)

        # Total thermal resistances [K/W]
        R_base_tot = R_base + R_base_ext
        R_side_tot = R_side + R_side_ext

        # U-value [W/K]
        U_tank = 2/R_base_tot + 1/R_side_tot 
        return U_tank


#%%
@dataclass
class AirSourceHeatPump:
    '''
    물리적 원리에 기반한 히트펌프 성능 계산 및 최적 운전점 탐색 클래스.
    '''
    def __init__(self,
                 refrigerant        = 'R410A',
                 disp_cmp           = 0.0005,
                 eta_cmp_isen       = 0.7,
                 eta_cmp_dV = 0.85,
                 A_iu       = 15.0,   # 응축기 전열 면적 [m2]
                 A_ou       = 20.0,   # 증발기 전열 면적 [m2]
                 U_coeff_iu = 100.0,  # 응축기 열전달 특성 계수
                 U_coeff_ou = 80.0,   # 증발기 열전달 특성 계수
                 dT_iu_ref = 5.0,
                 dT_ou_ref = 5.0,
                 T_ia    = 20.0,
                 Q_iu    = 6000,
                 ):
        '''
        히트펌프의 고정된 물리적 파라미터를 초기화합니다.

        Args:
            refrigerant (str): 사용할 냉매 이름 (CoolProp 형식).
            disp_cmp (float): 압축기 행정 체적 (1회전 당 흡입량) [m^3].
            eta_cmp_isen (float): 압축기 단열 효율. - 단열 효율은 압축 과정에서 발생하는 에너지 손실이 얼마나 적은가를 나타내는 지표
            eta_cmp_dV (float): 압축기 체적 효율. - 압축기가 한 번 회전할 때 이론적으로 빨아들일 수 있는 냉매량 대비, 실제로 얼마나 빨아들였는가를 나타내는 지표
            A_iu (float): 실내기 전열 면적 [m2].
            A_ou (float): 실외기 전열 면적 [m2].
            U_coeff_iu (float): 실내기 열전달 특성 계수.
            U_coeff_ou (float): 실외기 열전달 특성 계수.
            dT_iu_ref (float): 실내기 접근 온도차 (응축온도 - 실내온도) [K]. 
            dT_ou_ref (float): 실외기 접근 온도차 (실외온도 - 증발온도) [K].
            T_ia (float): 목표 실내 공기 온도 [°C].
            iu_fan (dict): 실내기 팬 데이터.
            ou_fan (dict): 실외기 팬 데이터.
            Q_iu (float): 실내기 목표 열 교환율 [W].
        '''
        
        self.ref = refrigerant
        self.disp_cmp  = disp_cmp 
        self.eta_cmp_isen = eta_cmp_isen 
        self.eta_cmp_dV = eta_cmp_dV
        
        self.A_iu = A_iu
        self.A_ou = A_ou
        
        self.U_coeff_iu = U_coeff_iu
        self.U_coeff_ou = U_coeff_ou
        
        self.dT_iu_ref = dT_iu_ref
        self.dT_ou_ref = dT_ou_ref
        self.T_ia = cu.C2K(self.T_ia)
        
        
        self.Q_iu = Q_iu # 실내기 목표 열 교환율
        self.mode = 'heating' if Q_iu < 0 else 'cooling'

        def _solve_for_fan_airflow(self, Q_target, T_air_in, T_ref, A, U_coeff):
            '''
            Args:
                Q_target (float): 목표 열교환량 [W]. (+) 냉매
                T_air_in (float): 공기 입구 온도 [°C].
                T_ref (float): 냉매 온도 [K].
                A (float): 열교환기 전열 면적 [m^2].
                U_coeff (float): 열교환기 열전달 특성 계수.
            Returns: 
                dV_fan (float): 필요 풍량 [m^3/s] 또는 None
            Description:
                목표 열교환율 (Q_target)을 만족시키기 위한 필요 풍량(dV_fan)을 수치적으로 계산한다.
                열교환율은 다음을 만족시켜야한다.
                1) 공기 측 에너지 공식: Q = c_a * dV_fan * rho_a * (T_air_in - T_air_out)
                2) 열교환기 공식: Q = U * A * LMTD
                2-1) 이때 U는 풍량에 따라 변하며, U ∝ dV_fan^0.8로 가정한다. 
                Q_target은 positive(+) 일때 냉매에 흡수되는 방향으로, negative(-) 일때 냉매에서 방출되는 방향으로 정의된다.
            '''
            
            # 절대온도 변환
            T_air_in = cu.C2K(T_air_in)
            
            # 열교환 방정식을 만족하는 dV_fan을 찾기 위한 오차 함수
            def error_function(dV_fan):
                if dV_fan <= 0: return 1e6 # 풍량이 0 이하인 경우 큰 오차 반환
                
                # 1. 공기 측 에너지 공식으로부터 공기 출구 온도 계산
                T_air_out = T_air_in - Q_target / (dV_fan * rho_a * c_a)
                
                # 2. 열교환기 공식으로부터 열교환량 계산
                # LMTD 계산
                delta_T1 = T_air_in - T_ref # T_air_in > T_ref 가정
                delta_T2 = T_air_out - T_ref # T_air_out > T_ref 가정
                
                if delta_T1 <= 0 or delta_T2 <= 0 or delta_T1 == delta_T2:
                    # 물리적으로 불가능하거나 LMTD 계산이 불가한 경우
                    return 1e6
                LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
                
                # 풍량에 따른 U값 계산 (U ∝ dV_fan^0.8 가정)
                '''
                Incropera & DeWitt의 'Fundamentals of Heat and Mass Transfer
                '''
                U = U_coeff * (dV_fan**0.8)
                
                Q_calculated = U * A * LMTD
                
                return Q_calculated - Q_target

            # 수치해석적 해법(Root-finding)으로 오차 함수가 0이 되는 dV_fan 탐색
            # root_scalar는 특정 함수의 결과값이 0이 되는 입력값 x (즉, 해(root))를 찾는 수치해석 함수
            try:
                sol = root_scalar(error_function, bracket=[0.01, 10.0], method='brentq')
                if sol.converged:
                    return sol.root
                else:
                    return None # 해를 찾지 못한 경우
            except ValueError:
                return None
    
    def calc_fan_power_from_parameters(
        P_max, 
        dV_max, 
        eta_max, 
        k_sys, 
        A_fan, 
        bep_ratio=0.75
        ):
        '''
        팬의 핵심 성능 파라미터와 시스템 저항 계수로부터
        운전점을 찾아내고, 해당 지점의 투입 전력과 속도를 계산합니다.

        Args:
            P_max (float): 팬의 최대 정압 (Shut-off Head) [Pa].
            dV_max (float): 팬의 최대 풍량 (Free Delivery) [m³/s].
            eta_max (float): 팬의 최대 전효율 [0.0 ~ 1.0].
            k_sys (float): 시스템 저항 계수 [Pa/(m³/s)²].
            A_fan (float): 팬의 토출 면적 [m²].
            bep_ratio (float, optional): 최대 효율점(BEP)이 발생하는 풍량의 비율. 
                                        기본값은 0.75 (최대 풍량의 75% 지점).

        Returns:
            dict: 운전점의 풍량, 정압, 속도, 투입 전력을 담은 딕셔너리.
                해를 찾지 못하면 None을 반환합니다.
        '''
        
        # --- 1. 성능 및 저항 곡선 함수 정의 ---
        def fan_performance_curve(dV):
            '''P_max와 dV_max를 이용해 팬 성능 곡선을 2차 포물선으로 근사'''
            if dV > dV_max: return 0
            return P_max * (1 - (dV / dV_max)**2)

        def system_resistance_curve(dV):
            '''시스템 저항 곡선'''
            return k_sys * dV**2

        # --- 2. 운전점(풍량) 탐색 ---
        # 오차 함수: P_fan(dV) - P_sys(dV) = 0
        def error_function(dV):
            if dV < 0: return 1e6
            return fan_performance_curve(dV) - system_resistance_curve(dV)

        try:
            sol = root_scalar(error_function, bracket=[0, dV_max], method='brentq')
            if sol.converged:
                dV_op = sol.root  # 운전점 풍량 (Operating flow rate)
            else:
                return None
        except ValueError:
            return None
        
        # --- 3. 운전점에서의 값 계산 ---
        # 운전점 압력
        P_op = system_resistance_curve(dV_op)

        # 운전점 효율 (최고 효율점을 갖는 포물선으로 근사)
        dV_bep = dV_max * bep_ratio # 최고 효율점 풍량
        # 정규화된 풍량 (최고점에서 1, 양 끝에서 0이 되도록)
        norm_dv = 1 - ((dV_op - dV_bep) / dV_bep)**2 if dV_op < dV_bep else 1 - ((dV_op - dV_bep) / (dV_max - dV_bep))**2
        eta_op = eta_max * max(0, norm_dv) # 효율 계산, 음수 방지

        # 팬 토출 속도
        velocity_op = dV_op / A_fan

        # 최종 투입 전력
        power_input = (dV_op * P_op) / eta_op if eta_op > 0 else float('inf')

        return {
            'operating_flow_rate_m3_s': dV_op,
            'operating_pressure_Pa': P_op,
            'operating_efficiency': eta_op,
            'operating_velocity_m_s': velocity_op,
            'required_power_W': power_input
        }
    
    def _calculate_cycle_performance(self, cmp_rps, T0):
        '''
        EX) 난방 모드 기준 사이클 다이어그램
        
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
        (저온/저압 액체+가스)                                   (고압 액체)
        '''
        
        # --- 1. 증발 및 응축 온도/압력 계산 ---
        if self.mode == 'heating':
            T0 = cu.C2K(T0)
            
            # --- 1. 증발 및 응축 온도/압력 계산 (역할 기반으로 명확화) ---
            # 난방 모드: 실내기 = 응축기, 실외기 = 증발기
            
            # 응축기(실내기) 온도/압력
            T_iu_ref = self.T_ia + self.dT_iu_ref
            P_iu_ref = CP.PropsSI('P', 'T', T_iu_ref, 'Q', 0, self.ref)

            # 증발기(실외기) 온도/압력
            T_ou_ref = T0 - self.dT_ou_ref
            P_ou_ref = CP.PropsSI('P', 'T', T_ou_ref, 'Q', 1, self.ref)

            # --- 2. 사이클의 각 지점(State 1, 2, 3, 4) 물성치 계산 ---
            # State 1: 압축기 입구 (저압의 포화 증기)
            #          증발기(실외기) 출구에 해당
            h1 = CP.PropsSI('H', 'P', P_ou_ref, 'Q', 1, self.ref)
            s1 = CP.PropsSI('S', 'P', P_ou_ref, 'Q', 1, self.ref)
            rho1 = CP.PropsSI('D', 'P', P_ou_ref, 'Q', 1, self.ref)

            # State 2: 압축기 출구 (고압의 과열 증기)
            #          저압(P_evap)에서 고압(P_cond)으로 압축
            h2_isen = CP.PropsSI('H', 'P', P_iu_ref, 'S', s1, self.ref) 
            h2 = h1 + (h2_isen - h1) / self.eta_cmp_isen
            
            # State 3: 응축기 출구 (고압의 포화 액체)
            #          응축기(실내기) 출구에 해당
            h3 = CP.PropsSI('H', 'P', P_iu_ref, 'Q', 0, self.ref)

            # State 4: 팽창밸브 출구 (저압의 액체+기체 혼합물)
            h4 = h3

            # --- 3. 성능 지표 계산 ---
            m_dot_ref = cmp_rps * self.disp_cmp * rho1 * self.eta_cmp_dV
            
            # 실내기(응축기) 방출 열량 (난방 능력) -> 음수(-)
            Q_iu = -(m_dot_ref * (h2 - h3))
            
            # 실외기(증발기) 흡수 열량
            Q_ou = m_dot_ref * (h1 - h4)
            
            # 압축기 사용 전력
            E_cmp = m_dot_ref * (h2 - h1)
            
        elif self.mode == 'cooling':
            T0 = cu.C2K(T0)
            
            # --- 1. 증발 및 응축 온도/압력 계산 (이 부분은 올바름) ---
            # 실외기(응축기) 온도/압력
            T_ou_ref = T0 + self.dT_ou_ref
            P_ou_ref = CP.PropsSI('P', 'T', T_ou_ref, 'Q', 0, self.ref)

            # 실내기(증발기) 온도/압력
            T_iu_ref = self.T_ia - self.dT_iu_ref
            P_iu_ref = CP.PropsSI('P', 'T', T_iu_ref, 'Q', 1, self.ref)

            # --- 2. 사이클의 각 지점(State 1, 2, 3, 4) 물성치 계산 (수정된 부분) ---
            # State 1: 압축기 입구 (저압의 포화 증기)
            #          실내기(증발기) 출구에 해당
            h1 = CP.PropsSI('H', 'P', P_iu_ref, 'Q', 1, self.ref)
            s1 = CP.PropsSI('S', 'P', P_iu_ref, 'Q', 1, self.ref)
            rho1 = CP.PropsSI('D', 'P', P_iu_ref, 'Q', 1, self.ref)

            # State 2: 압축기 출구 (고압의 과열 증기)
            #          저압(P_iu_ref)에서 고압(P_ou_ref)으로 압축
            h2_isen = CP.PropsSI('H', 'P', P_ou_ref, 'S', s1, self.ref) 
            h2 = h1 + (h2_isen - h1) / self.eta_cmp_isen
            
            # State 3: 응축기 출구 (고압의 포화 액체)
            #          실외기(응축기) 출구에 해당
            h3 = CP.PropsSI('H', 'P', P_ou_ref, 'Q', 0, self.ref)
            
            # State 4: 팽창밸브 출구 (저압의 액체+기체 혼합물)
            h4 = h3

            # --- 3. 성능 지표 계산 (변수명 통일) ---
            m_dot_ref = cmp_rps * self.disp_cmp * rho1 * self.eta_cmp_dV
            
            # 실내기(증발기) 흡수 열량 (냉방 능력) -> 양수(+)
            Q_iu = m_dot_ref * (h1 - h4)
            
            # 실외기(응축기) 방출 열량
            Q_ou = m_dot_ref * (h2 - h3)
            
            # 압축기 사용 전력
            E_cmp = m_dot_ref * (h2 - h1)
        else:
            raise ValueError('Invalid mode. Mode should be either "heating" or "cooling".')

        ##########################################################################
        '''
        팬 사용 전력 계산
        공기와 열교환기가 교환하는 과정에서 두 교환된 열교환율이 같다는 가정으로, 연립방정식을 풀어야함.
        또한 열교환기 측 총괄열전달계수는 팬 풍량에 따라 변하는 변수이므로, 팬 풍량에 따른 열교환기 총괄열전달계수를 구하는 과정이 필요함.
        Q = U * A * LMTD, U = f(dV_fan)           - (열교환기 측)
        Q = c_a * V_dot_air (T_a_in - T_a_out)    - (공기 측)
        '''
        dV_iu_fan = self._solve_for_fan_airflow(Q_iu, self.T_ia, T_iu_ref, self.A_iu, self.U_coeff_iu)
        dV_ou_fan = self._solve_for_fan_airflow(Q_ou, cu.K2C(T0), T_ou_ref, self.A_ou, self.U_coeff_ou)
        E_iu_fan = self.fan_system.get_power(self.indoor_fan, dV_iu_fan) 
        E_ou_fan = self.fan_system.get_power(self.outdoor_fan, dV_ou_fan)
        ##########################################################################
        
        E_tot = E_cmp + E_iu_fan + E_ou_fan
        
        # COP (Coefficient of Performance -> system energy efficiency)
        cop = Q_iu / E_tot if E_tot > 0 else 0

        return {
            'Q_iu': Q_iu, # W
            'Q_ou': Q_ou, # W
            'E_cmp': E_cmp, # W
            'E_iu_fan': E_iu_fan, # W
            'E_ou_fan': E_ou_fan, # W
            'E_tot': E_tot,
            'cop': cop,
            'm_dot_ref_kg_s': m_dot_ref,
            'T_ou_ref': T_ou_ref,
            'P_ou_ref_kPa': P_ou_ref * cu.Pa2kPa,
            'T_iu_ref': T_iu_ref,
            'P_iu_ref_kPa': P_iu_ref * cu.Pa2kPa,
        }

    def find_optimal_operation(self, required_heating_load_kW, T0):
        '''
        주어진 난방 부하와 외기온도 조건에서 총 전력사용를 최소화하는
        압축기 및 팬 운전 조건을 찾습니다.

        Args:
            required_heating_load_kW (float): 요구되는 난방 부하 [kW].
            T0 (float): 실외 공기 온도 [°C].

        Returns:
            dict: 최적화 결과 또는 에러 메시지.
        '''
        # 최적화 변수: x[0] = 압축기 회전수(rps), x[1] = 팬 풍량(m^3/s)
        
        # 1. 목적 함수: 총 전력 사용량 (최소화 대상)
        def objective(x):
            comp_speed, fan_airflow = x
            perf = self._calculate_cycle_performance(comp_speed, fan_airflow, T0)
            return perf["E_tot"]

        # 2. 제약 조건: 계산된 난방 능력이 요구 부하와 같아야 함
        def constraint(x):
            comp_speed, fan_airflow = x
            perf = self._calculate_cycle_performance(comp_speed, fan_airflow, T0)
            # solver가 0을 만족하는 해를 찾으므로 (계산값 - 목표값) 형태로 반환
            return perf["Q_iu"] - required_heating_load_kW

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
                optimal_comp_speed, optimal_fan_airflow, T0
            )
            
            # 보기 쉽게 결과 정리
            output = {
                'success': True,
                'message': '최적 운전점을 찾았습니다.',
                'required_load_kW': required_heating_load_kW,
                'T0': T0,
                'optimal_compressor_speed_rps': round(optimal_comp_speed, 2),
                'optimal_compressor_speed_rpm': round(optimal_comp_speed * 60, 0),
                'optimal_dV_fan': round(optimal_fan_airflow, 3),
                'performance': {
                    'Calculated_Q_iu': round(final_performance["Q_iu"], 3),
                    'COP': round(final_performance["cop"], 3),
                    'E_tot': round(final_performance["E_tot"], 3),
                    'E_cmp': round(final_performance["E_cmp"], 3),
                    'E_fan': round(final_performance["E_fan"], 3),
                    'Evaporating_Temp_C': round(final_performance["T_ou_ref"], 2),
                    'Condensing_Temp_C': round(final_performance["T_iu_ref"], 2),
                }
            }
            return output
        else:
            return {
                'success': False,
                'message': f'최적화에 실패했습니다: {result.message}'
            }


@dataclass
class GroundSourceHeatPumpBoiler2:
    '''
    물리적 원리에 기반한 지열워 히트펌프 성능 계산 및 최적 운전점 탐색 클래스.
    '''
    def __init__(
        self,
        
        # 냉매 종류
        refrigerant  = 'R410A',
        
        # 압축기 관련 파라미터
        disp_cmp     = 0.0005,
        eta_cmp_isen = 0.7,
        eta_cmp_dV   = 0.85,
        
        # 온도 관련 파라미터
        T_f_bh_in    = 15.0,
        Tg           = 15.0,
        
        # 열교환기 관련 파라미터
        UA_HX_tank       = 500, # W/K
        UA_HX_water_loop = 500, # W/K
    
        ######################################################
        # # Tank size [m]
        # r0 = 0.2,
        # H = 0.8,
        
        # # Tank layer thickness [m]
        # x_shell = 0.01,
        # x_ins   = 0.10,
    
        # # Tank thermal conductivity [W/mK]
        # k_shell = 25, 
        # k_ins   = 0.03, 

        # # Overall heat transfer coefficient [W/m²K]
        # h_o = 15, 
        ######################################################
    
        # Borehole parameters
        D_b = 0, # Borehole depth [m]
        H_b = 200, # Borehole height [m]
        r_b = 0.08, # Borehole radius [m]
        R_b = 0.108, # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        dV_f = 24, # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        k_g   = 2.0,
        c_g   = 800,
        rho_g = 2000,

        # Pump of ground heat exchanger
        E_pmp  = 200,
        ):
        '''
        히트펌프의 고정된 물리적 파라미터를 초기화합니다.

        Args:
            refrigerant (str)         : 사용할 냉매 이름 (CoolProp 형식).
            disp_cmp (float)          : 압축기 행정 체적 (1회전 당 흡입량) [m^3].
            eta_cmp_isen (float)      : 압축기 단열 효율. - 단열 효율은 압축 과정에서 발생하는 에너지 손실이 얼마나 적은가를 나타내는 지표
            eta_cmp_dV (float): 압축기 체적 효율. - 압축기가 한 번 회전할 때 이론적으로 빨아들일 수 있는 냉매량 대비, 실제로 얼마나 빨아들였는가를 나타내는 지표
            dT_ref_tank (float)       : 저탕조 접근 온도차 (응축온도 - 저탕조 온도) [K]. 
            dT_ref_HX (float)         : 열교환기 접근 온도차 (지중온도 - 증발온도) [K].
            T_w_tank (  float)         : 저탕조 목표 온도 [°C].
            T_f_HX (float)            : 지중온도 [°C].
            T0 (float or None)        : 초기 외기 온도 [°C].
                                        None인 경우, 시뮬레이션 시간 동안 외기 온도를 0°C로 가정.
            Tg (float)                : 지중온도 [°C].
            Q_ref_tank (array or None): 저탕조 목표 열 교환율 [W].
                                        None인 경우, 시뮬레이션 시간 동안 열 교환율을 0W로 가정.
                                        배열인 경우, 길이가 시뮬레이션 스텝 수와 일치해야 함.
            dt_s (int)                : 시뮬레이션 시간 간격 [초].
            time_h (int)              : 시뮬레이션 총 시간 [시간].
        '''

        self.ref          = refrigerant
        self.disp_cmp     = disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        self.eta_cmp_dV   = eta_cmp_dV
        
        self.Tg        = Tg
        self.T_f_bh_in = T_f_bh_in
        
        self.UA_HX_tank       = UA_HX_tank
        self.UA_HX_water_loop = UA_HX_water_loop
        
        # self.r0 = r0
        # self.H = H
        
        # self.x_shell = x_shell
        # self.x_ins   = x_ins
        # self.k_shell = k_shell
        # self.k_ins   = k_ins
        # self.h_o     = h_o
        
        self.D_b = D_b
        self.H_b = H_b
        self.r_b = r_b
        self.R_b = R_b
        
        self.dV_f  = dV_f  # L/min -> m^3/s
        self.k_g   = k_g
        self.c_g   = c_g
        self.alp_g = k_g / (c_g * rho_g)
        self.rho_g = rho_g
        self.E_pmp = E_pmp
        
        # Unit conversion
        self.Tg    = cu.C2K(self.Tg)
        self.T_f_bh_in = cu.C2K(self.T_f_bh_in)
        self.dV_f  = self.dV_f * cu.L2m3/cu.m2s  # L/min -> m^3/s
        
        self.Q_LOAD_OFF_ATOL = 500.0     # [W] 이하면 완전 OFF
        
    def _off_result(self, T_w_tank):
        """장치 OFF 상태의 결과 패키징(모든 열량/전력 0, 펌프도 OFF 가정)."""
        T_w_K = cu.C2K(T_w_tank)
        T_f_in = self.T_f_bh_in

        # 포화점은 '참조값'으로만 계산(그려도 되고 안 그려도 됨)
        try:
            P1 = CP.PropsSI('P', 'T', T_f_in, 'Q', 1, self.ref)
            P3 = CP.PropsSI('P', 'T', T_w_K,  'Q', 0, self.ref)
            h1 = CP.PropsSI('H', 'P', P1, 'Q', 1, self.ref); s1 = CP.PropsSI('S', 'P', P1, 'Q', 1, self.ref)
            h3 = CP.PropsSI('H', 'P', P3, 'Q', 0, self.ref); s3 = CP.PropsSI('S', 'P', P3, 'Q', 0, self.ref)
        except Exception:
            # 혹시 범위 밖이면 NaN으로
            P1=P3=h1=h3=s1=s3=np.nan

        result = {
            'Q_ref_tank': 0.0, 'Q_ref_HX': 0.0,
            'Q_LMTD_tank': 0.0, 'Q_LMTD_HX': 0.0,
            'Q_load': 0.0,
            'E_cmp': 0.0, 'E_cmp_eff': 0.0,
            'E_pmp_eff': 0.0,               # 펌프도 OFF
            'cmp_rps': 0.0, 'm_dot_ref': 0.0,
            'T1': T_f_in, 'T2': T_f_in, 'T3': T_w_K, 'T4': T_w_K,
            'T_f_bh_in': T_f_in, 'T_f_bh_out': T_f_in,
            'P1': P1, 'P2': P3, 'P3': P3, 'P4': P1,
            'h1': h1, 'h2': h1, 'h3': h3, 'h4': h3,
            's1': s1, 's2': s1, 's3': s3, 's4': s3,
            'is_on': False,
        }
        return result
    
    def _calculate_cycle_performance(self, dT_ref_tank, dT_ref_HX, T_w_tank, Q_load):
        '''
        EX) 난방 모드 기준 사이클 다이어그램
        
        주어진 운전 조건(압축기/팬 속도, 외기온도)에서 사이클 성능을 계산하는 내부 함수.
        (저온/저압 가스)                                (고온/고압 가스)
        (1) -------------------- [ 압축기 ] --------------------> (2)
        ^                                                        |
        |                                                        v
      [열교환기]                                                [저탕조]
      (열 흡수 ❄️)                                           (열 방출 🔥)
        ^                                                        |
        |                                                        v
        (4) <----------------- [ 팽창밸브 ] <------------------- (3)
        (저온/저압 액체+가스)                                   (고압 액체)
        '''
        
        T_w_tank = cu.C2K(T_w_tank)
        T_f_bh_in = self.T_f_bh_in
        
        # --- 1. 증발 및 응축 온도/압력 계산 ---
        # 난방 모드: 실내기 = 응축기, 실외기 = 증발기
        
        # 응축기(실내기) 온도/압력
        T3 = T_w_tank + dT_ref_tank # T3 
        P3 = CP.PropsSI('P', 'T', T3, 'Q', 0, self.ref) # P3

        # 증발기(열교환기) 온도/압력
        T1 = T_f_bh_in - dT_ref_HX
        P1 = CP.PropsSI('P', 'T', T1, 'Q', 1, self.ref)

        # --- 2. 사이클의 각 지점(State 1, 2, 3, 4) 물성치 계산 ---
        # State 1: 압축기 입구 (저압의 포화 증기)
        h1   = CP.PropsSI('H', 'P', P1, 'Q', 1, self.ref)
        s1   = CP.PropsSI('S', 'P', P1, 'Q', 1, self.ref)
        rho1 = CP.PropsSI('D', 'P', P1, 'Q', 1, self.ref)

        # State 2: 압축기 출구 (고압의 과열 증기)
        h2_isen = CP.PropsSI('H', 'P', P3, 'S', s1, self.ref) 
        h2 = h1 + (h2_isen - h1) / self.eta_cmp_isen
        
        T2 = CP.PropsSI('T', 'P', P3, 'H', h2, self.ref)
        P2 = P3
        s2 = CP.PropsSI('S', 'P', P3, 'H', h2, self.ref)
        
        # State 3: 응축기 출구 (고압의 포화 액체)
        h3 = CP.PropsSI('H', 'P', P3, 'Q', 0, self.ref)
        s3 = CP.PropsSI('S', 'P', P3, 'Q', 0, self.ref)

        # State 4: 팽창밸브 출구 (저압의 액체+기체 혼합물)
        h4 = h3
        P4 = P1
        T4 = CP.PropsSI('T', 'P', P1, 'H', h4, self.ref)
        s4 = CP.PropsSI('S', 'P', P1, 'H', h4, self.ref)

        # --- 3. 성능 지표 계산 ---
        # Q_load를 만족시키기 위해 필요한 냉매 유량(m_dot_ref)을 역산
        # Q_load와 (h3 - h2)는 모두 음수이므로 m_dot_ref은 양수가 됨
        h3 - h2
        if (h3 - h2) == 0: return None
        m_dot_ref = Q_load / (h3 - h2) # (Q_load < 0, (h3 - h2) < 0) -> m_dot_ref > 0 [kg/s]
        
        # 계산된 m_dot_ref을 만들기 위해 필요한 압축기 회전수(cmp_rps)를 역산
        denominator = self.disp_cmp * rho1 * self.eta_cmp_dV
        if denominator == 0: return None
        cmp_rps = m_dot_ref / denominator # [1/s]
        
        # 계산된 값들로 나머지 성능 지표 계산
        Q_ref_tank = m_dot_ref * (h3 - h2) # 이 값은 Q_load와 거의 동일
        Q_ref_HX   = m_dot_ref * (h1 - h4)
        E_cmp      = m_dot_ref * (h2 - h1)
        
        # --- 4. LMTD 기반 열량 계산 (현실 제약 조건) ---
        # 저탕조 측 (응축기)
        delta_T1_tank = T2 - T_w_tank
        delta_T2_tank = T3 - T_w_tank
        # 0 또는 음수 온도차 방지
        if delta_T1_tank <= 1e-6 or delta_T2_tank <= 1e-6 or abs(delta_T1_tank - delta_T2_tank) < 1e-6:
             Q_LMTD_tank = -np.inf # 물리적으로 불가능한 경우 패널티
        else:
             LMTD_tank = (delta_T1_tank - delta_T2_tank) / np.log(delta_T1_tank / delta_T2_tank)
             Q_LMTD_tank = self.UA_HX_tank * LMTD_tank

        # 지중열 측 (증발기) - 대향류(Counter-flow) 모델 수정
        m_dot_f = self.dV_f * rho_w # __init__에서 계산해도 됨
        T_f_bh_out = T_f_bh_in + Q_ref_HX / (c_w * rho_w * self.dV_f) # 지중열 유입구 온도 + (열교환율 / (비열 * 밀도 * 유량)

        delta_T1_HX = T_f_bh_in - T1
        delta_T2_HX = T_f_bh_out - T4
        # 0 또는 음수 온도차 방지
        if delta_T1_HX <= 1e-6 or delta_T2_HX <= 1e-6 or abs(delta_T1_HX - delta_T2_HX) < 1e-6:
            Q_LMTD_HX = np.inf # 물리적으로 불가능한 경우 패널티
        else:
            LMTD_HX = (delta_T1_HX - delta_T2_HX) / np.log(delta_T1_HX / delta_T2_HX)
            Q_LMTD_HX = self.UA_HX_water_loop * LMTD_HX
        
        result = {
            'is_on'    : True,
            
            'Q_ref_tank' : Q_ref_tank,    # W
            'Q_ref_HX'   : Q_ref_HX,      # W
            'Q_LMTD_tank': Q_LMTD_tank,   # W
            'Q_LMTD_HX'  : Q_LMTD_HX,     # W
            
            'Q_load'   : Q_load,      # W
            'E_cmp'    : E_cmp,       # W
            'cmp_rps'  : cmp_rps,     # rps
            'm_dot_ref': m_dot_ref,   # kg/s
            
            'T1': T1,   # K
            'T2': T2,   # K
            'T3': T3,   # K
            'T4': T4,   # K
            
            'T_f_bh_in' : T_f_bh_in,    # K
            'T_f_bh_out': T_f_bh_out,   # K
            
            'P1': P1,   # kPa
            'P2': P2,   # kPa
            'P3': P3,   # kPa
            'P4': P4,   # kPa
            
            'h1': h1, # J/kg
            'h2': h2, # J/kg
            'h3': h3, # J/kg
            'h4': h4, # J/kg
            
            's1': s1, # J/kgK
            's2': s2, # J/kgK
            's3': s3, # J/kgK
            's4': s4, # J/kgK
        }
        self.__dict__.update(result)
        return result

    def _find_ref_loop_optimal_operation(self, T_w_tank, Q_load):
        '''
        dT에 따라서, 결국 LMTD를 만족하는 dT들의 조합이 존재한다.
        근데 이때 dT(dT_ref_tank, dT_ref_HX)에 따라서, Ecmp가 최소가 되어야하므로,
        dT(dT_ref_tank, dT_ref_HX)에 따른 냉매의 유량(m_dot_ref) 변화, dT에 따른 h2-h1의 변화가 복합적으로 Ecmp를 결정하므로
        어떠한 dT의 조합들에 대해서 E_cmp를 최소화시키는 운전점이 존재하고 그 지점을 찾는 것이다.
s
        Args:
            T_w_tank (float): 저탕조 목표 온도 [°C].
            Q_load (float): 저탕조 목표 열 교환율 [W]. (난방 부하, 음수 값)

        Returns:
            dict: 최적화 결과 또는 에러 메시지.
        '''
        # 최적화 변수: x[0] = 압축기 회전수(rps), x[1] = 냉매 저탕조 온도차(K), x[2] = 냉매-열교환기 온도차(K)
        
        # --- 0) OFF/소부하 처리 ---
        Q_req = float(Q_load)
        if abs(Q_req) <= self.Q_LOAD_OFF_ATOL:
            return self._off_result(T_w_tank)
        
        # 1. 목적 함수: 총 전력 사용량 (최소화 대상)
        
        def objective(x):
            dT_ref_HX, dT_ref_tank = x
            perf = self._calculate_cycle_performance(
                dT_ref_tank=dT_ref_tank, dT_ref_HX=dT_ref_HX,
                T_w_tank=T_w_tank, Q_load=Q_load,
            )
            return perf["E_cmp"]

        # 🎯 제약 조건 함수들 정의
        def constraint_tank(x):
            '''
            Q_LMTD_tank: 주어진 T2, T3와 T_w_tank에 기반해 계산된 냉매-저탕조 온수 열 교환율 [W]
            Q_ref_tank: 냉매 사이클 계산으로부터 얻어진 냉매-저탕조 열 교환율 [W]
            제약 조건: Q_LMTD_tank >= |Q_ref_tank| ↔  Q_LMTD_tank + Q_ref_tank >= 0
            '''
            dT_ref_HX, dT_ref_tank = x
            perf = self._calculate_cycle_performance(dT_ref_tank, dT_ref_HX, T_w_tank, Q_load)
            # Q_ref_tank는 난방에서 음수이므로, |Q_ref_tank| = -Q_ref_tank
            return perf['Q_LMTD_tank'] + perf['Q_ref_tank'] # (양수) + (음수)

        def constraint_hx(x):
            dT_ref_HX, dT_ref_tank = x
            perf = self._calculate_cycle_performance(dT_ref_tank, dT_ref_HX, T_w_tank, Q_load)
            return perf['Q_LMTD_HX'] - perf['Q_ref_HX'] # (양수) - (양수)
        
        bounds = [(0.1, 30.0), (0.1, 30.0)]
        initial_guess = [5, 5]

        # 제약 조건 리스트 생성
        cons = [
                {'type': 'eq', 'fun': constraint_tank}, # ineq: Q_LMTD_tank - |Q_ref_tank| >= 0
                {'type': 'eq', 'fun': constraint_hx},
            ]

        # 최적화 실행 (constraints 인자 추가)
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=cons, options={'disp': False})

        if result.success:
            optimal_dT_ref_HX, optimal_dT_ref_tank = result.x
            final_performance = self._calculate_cycle_performance(
                dT_ref_tank=optimal_dT_ref_tank, dT_ref_HX=optimal_dT_ref_HX,
                T_w_tank=T_w_tank, Q_load=Q_load
            )
            return final_performance
        else:
            # 최적화 실패 시, 실패 원인 분석
            fail_reason = result.message  # 기본적인 실패 메시지

            # result.status 코드를 통해 좀 더 구체적인 원인 파악
            # (scipy 문서 참조: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult )
            if result.status == 1:
                fail_reason += " (Iteration limit reached - 반복 횟수 초과)"
            elif result.status == 2:
                fail_reason += " (Precision loss - 정밀도 손실)"
            elif result.status == 3:
                fail_reason += " (Objective/constraint function error - 함수 계산 오류)"
            elif result.status == 4:
                fail_reason += " (Iteration limit reached - 반복 횟수 초과)" # SLSQP에선 1과 4가 비슷하게 쓰일 수 있음
            elif result.status == 8:
                 fail_reason += " (Infeasible constraints - 제약 조건 만족 불가)"
            elif result.status == 9:
                 fail_reason += " (Iteration limit reached - 반복 횟수 초과)" # SLSQP 특정

            # 최종 지점에서의 제약 조건 값 확인 (어떤 제약이 위반되었는지 추정)
            try:
                final_x = result.x if hasattr(result, 'x') else initial_guess
                c_tank_val = constraint_tank(final_x)
                c_hx_val = constraint_hx(final_x)
                fail_reason += f"\n  - 최종 제약 조건 값: Tank={c_tank_val:.3f}, HX={c_hx_val:.3f}"
                if c_tank_val < -1e-6 or c_hx_val < -1e-6: # 부등식 제약 조건 위반 (0보다 작음)
                     fail_reason += " (제약 조건 위반 가능성 높음)"
            except Exception as e:
                fail_reason += f"\n  - 최종 제약 조건 확인 중 오류 발생: {e}"

            print(f'최적화에 실패했습니다:\n  - 원인: {fail_reason}')
            return None

    def plot_cycle_diagrams(self, result, save_path=None):
        '''
        계산된 사이클 상태(1,2,3,4)를 바탕으로 p-h, T-h 선도를 그립니다.
        '''
        # colors
        color1 = 'dm.blue5'
        color2 = 'dm.red5'
        color3 = 'dm.black'

        ymin1, ymax1, yint1 = 0, 10**4, 0
        ymin2, ymax2, yint2 = -20, 120, 20
        xmin, xmax, xint = 0, 500, 100

        # --- 임계/포화 데이터 준비 ---
        # (CoolProp 순서는 PropsSI('키', 유체명) 입니다)
        T_critical = cu.K2C(CP.PropsSI('Tcrit',  self.ref))
        P_critical = CP.PropsSI('Pcrit',  self.ref) / 1000  # kPa (참고용, 여기선 미사용)

        temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', self.ref)) + 1, T_critical, 200)
        h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, self.ref) / 1000 for T in temps]
        h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, self.ref) / 1000 for T in temps]
        p_sat = [CP.PropsSI('P', 'T', cu.C2K(T), 'Q', 0, self.ref) / 1000 for T in temps]

        # 상태값(kPa, kJ/kg, °C)
        p = np.array([result[f'P{i}'] for i in range(1, 5)])*cu.Pa2kPa
        h = np.array([result[f'h{i}'] for i in range(1, 5)])*cu.J2kJ
        T = np.array([result[f'T{i}'] for i in range(1, 5)]); T = cu.K2C(T)

        # 사이클 경로(닫기)
        h_cycle = np.concatenate([h, h[:1]])
        p_cycle = np.concatenate([p, p[:1]])
        T_cycle = np.concatenate([T, T[:1]])

        # --- Figure & Axes ---
        LW = np.arange(0.5, 3.0, 0.25)
        nrows, ncols = 1, 2
        fig, axes = plt.subplots(figsize=(dm.cm2in(16), dm.cm2in(7)), nrows=nrows, ncols=ncols)
        ax = axes.flatten()
        # 축별 메타데이터(인덱스로 접근)
        xlabels = ["Enthalpy [kJ/kg]", "Enthalpy [kJ/kg]"]
        ylabels = ["Pressure (log scale) [kPa]", "Temperature [°C]"]
        yscales = ["log", "linear"]
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
        if save_path is not None:
            plt.savefig(save_path, dpi=600)
        dm.save_and_show(fig)

   
