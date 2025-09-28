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
dm.use_style('dmpl_light')

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]
g = 9.81         # 중력가속도 [m/s²]
beta = 2.07e-4   # 물의 체적팽창계수 [1/K] (약 20°C 기준)

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
        - h_c_w (float): 열전달 계수 [W/m²K]
    🔹 Example
        ```
        h_c_w = compute_natural_convection_h_cp(T_s, T_inf, L)
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
    h_c_w = Nu_L * k_air / L  # [W/m²K]
    
    return h_c_w

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

def calc_boussinesq_mixing_flow(T_upper, T_lower, A, dz, C_d=0.1):
    """
    두 인접 노드 간의 부시네스크 근사에 기반한 혼합 유량을 계산합니다.
    혼합은 하단 노드의 온도가 상단 노드보다 높아 중력적으로 불안정할 때만 발생합니다.

    Parameters:
    -----------
    T_upper : float
        상단 노드의 온도 [K]
    T_lower : float
        하단 노드의 온도 [K]
    A : float
        탱크 단면적 [m²]
    dz : float
        노드 높이 [m]
    C_d : float, optional
        유량 계수 (경험적 상수), 기본값 0.1

    Returns:
    --------
    float
        두 노드 간 교환되는 체적 유량 [m³/s]
    """
    if T_lower > T_upper:
        # 하단이 더 따뜻하면 (밀도가 낮으면) 불안정하여 혼합 발생
        delta_T = T_lower - T_upper
        Q_mix = C_d * A * math.sqrt(g * beta * delta_T * dz)
        return Q_mix
    else:
        # 안정적인 상태에서는 혼합 없음
        return 0.0

def calc_UA_tank_arr(r0, x_shell, x_ins, k_shell, k_ins, H, N, h_w, h_o):
    """
    Overall heat-loss UA per vertical segment of a cylindrical tank (radial through side;
    planar through bottom/top). Side applies to all nodes; bottom/top add in parallel for node 1 and N.

    Calculate the overall heat transfer coefficient (U-value) of a cylindrical tank.
    Parameters:
    r0 : Inner radius of the tank [m]
    x_shell : Thickness of the tank shell [m]
    x_ins : Thickness of the insulation layer [m]
    k_shell : Thermal conductivity of the tank shell material [W/mK]
    k_ins : Thermal conductivity of the insulation material [W/mK]
    H : Height of the tank [m]
    N : Number of segments 
    h_w : Internal convective heat transfer coefficient [W/m²K]
    h_o : External convective heat transfer coefficient [W/m²K]
    Returns:
    UA_arr : Array of overall heat transfer coefficients for each segment [W/K]
    """
    dz = H / N
    r1 = r0 + x_shell
    r2 = r1 + x_ins

    # --- Areas ---
    # Side (per segment)
    A_side_in_seg  = 2.0 * math.pi * r0 * dz   # inner wetted area (for h_w)
    A_side_out_seg = 2.0 * math.pi * r2 * dz   # outer area (for h_o)
    # Bases (single discs)
    A_base_in  = math.pi * r0**2               # internal disc area (for h_w)
    A_base_out = math.pi * r2**2               # external disc area (for h_o)

    # --- Side: convection (in/out) + cylindrical conduction (shell + insulation) ---
    # Conduction (cylindrical) per segment
    R_side_cond_shell = math.log(r1 / r0) / (2.0 * math.pi * k_shell * dz)
    R_side_cond_ins   = math.log(r2 / r1) / (2.0 * math.pi * k_ins   * dz)
    R_side_cond = R_side_cond_shell + R_side_cond_ins  # [K/W]

    R_side_w   = 1.0 / (h_w * A_side_in_seg)          # [K/W]
    R_side_ext = 1.0 / (h_o * A_side_out_seg)         # [K/W]
    R_side_tot = R_side_w + R_side_cond + R_side_ext  # [K/W]  (series)

    # --- Bottom/Top discs: convection (in/out) + planar conduction (shell + insulation) ---
    # 권장: 각 층의 면적을 구분하여 직렬 합
    R_base_cond_shell = x_shell / (k_shell * A_base_in)   # [K/W]  (inner metal plate)
    R_base_cond_ins   = x_ins   / (k_ins   * A_base_out)  # [K/W]  (outer insulation plate)
    R_base_cond = R_base_cond_shell + R_base_cond_ins

    R_base_w   = 1.0 / (h_w * A_base_in)   # [K/W]
    R_base_ext = 1.0 / (h_o * A_base_out)  # [K/W]
    R_base_tot = R_base_w + R_base_cond + R_base_ext  # [K/W] (series through the base)

    # --- Equivalent node-to-ambient resistances ---
    # Middle nodes: side only
    R_mid = R_side_tot

    # Node 1 (bottom) and Node N (top): side || base
    R_end = (R_side_tot * R_base_tot) / (R_side_tot + R_base_tot)  # [K/W] (parallel)

    R_arr = np.array([R_end] + [R_mid]*(N-2) + [R_end], dtype=float)
    UA_arr = 1.0 / R_arr  # [W/K]
    return UA_arr


# Re-run after reset: build and execute the TDMA-based stratified tank demo

def TDMA(a,b,c,d) -> np.ndarray:
    """
    TDMA (Tri-Diagonal Matrix Algorithm)를 사용하여 온도를 업데이트합니다.
    
    Reference: https://doi.org/10.1016/j.ijheatmasstransfer.2017.09.057 [Appendix B - Eq.(B7)]
    
    만약 boundary condition이 None이 아닌 경우, 각각 추가된 최좌측, 최우측 열저항에 종합 열저항을 추가하여 계산하고 이를 바탕으로 TDMA 알고리즘을 적용함.
    
    즉, 대류 경계층을 boundary layer 함수를 통해 지정한 경우 Construction의 표면온도를 계산할 때, 정상상태를 가정하에 경계층 열저항을 고려하여 표면온도를 다시 구해줘야함
    
    Parameters:
    -----------
    a : np.ndarray
        하부 대각선 요소 (길이 N-1)
    b : np.ndarray
        주 대각선 요소 (길이 N)
    c : np.ndarray
        상부 대각선 요소 (길이 N-1)
    d : np.ndarray
        우변 벡터 (길이 N)
    Returns:
    --------
    np.ndarray
        다음 시간 단계의 온도 배열
    """
    n = len(b)

    A_mat = np.zeros((n, n))
    np.fill_diagonal(A_mat[1:], a[1:])
    np.fill_diagonal(A_mat, b)
    np.fill_diagonal(A_mat[:, 1:], c[:-1])
    A_inv = np.linalg.inv(A_mat)

    T_new = np.dot(A_inv, d).flatten() # Flatten the result to 1D array
    return T_new

def _add_loop_advection_terms(a, b, c, d, in_idx, out_idx, G_loop, T_loop_in):
    """
    지정 구간(in_idx -> out_idx)으로 흐르는 강제 대류를 TDMA 계수(a,b,cU,d)에 더함.
    - 인덱스는 0-based (노드 1 -> idx 0).
    - 방향: in_idx > out_idx 이면 '상향'(아래→위), 반대면 '하향'(위→아래).
    """
    if G_loop <= 0 or in_idx == out_idx:
        return

    if in_idx > out_idx:
        # 상향: in(N쪽) -> ... -> out(1쪽)
        # inlet 노드
        b[in_idx] += G_loop
        a[in_idx] -= G_loop              # 진행 이웃(i-1) 계수에 -G
        d[in_idx] += G_loop * T_loop_in  # 유입 스트림 온도
        # 경로 내부 노드 (out_idx+1 .. in_idx-1)
        for k in range(in_idx - 1, out_idx, -1):
            b[k] += G_loop
            c[k] -= G_loop              # 유입측 이웃(아래, k+1)에 -G
        # outlet 노드: out_idx -> outflow 경계 (추가 없음)

    else:
        # 하향: in(1쪽) -> ... -> out(N쪽)
        b[in_idx] += G_loop
        c[in_idx] -= G_loop             # 진행 이웃(i+1) 계수에 -G
        d[in_idx] += G_loop * T_loop_in
        for k in range(in_idx + 1, out_idx):
            b[k] += G_loop
            a[k] -= G_loop               # 유입측 이웃(위, k-1)에 -G
    # outlet 노드: out_idx -> outflow 경계 (추가 없음)
    
class StratifiedTankTDMA:
    '''
    To do: 탱크와 연결된 순환 루프 추가시 각 a, b, c, d 계수에 대한 수정 필요.
    현재는 탱크 상단 하단에서 동일한 물만 유입/유출되는 경우만 고려.
    Parameters:
    -----------
    H : float
        탱크 높이 [m]
    D : float
        탱크 직경 [m]
    N : int
        탱크를 분할하는 노드 수
    UA_arr : np.ndarray
        각 노드에 대한 전체 열전달 계수 배열 [W/K]
    Methods:
    --------
    step(T, dt, T_in, dV, T_amb, Q_heater
    주어진 시간 간격 dt 동안 탱크의 온도를 업데이트합니다.
    Parameters:
    -----------
    T : np.ndarray
        현재 노드 온도 배열 [K]
    dt : float
        시간 간격 [s]
    T_in : float
        유입수 온도 [K]
    dV : float
        유입/유출되는 물의 부피 [m³/s]
    T_amb : float
        주변 온도 [K]
    Q_heater_node : int or None, optional
        히터가 설치된 노드 번호 (1부터 N까지), 기본값은 None (히터 없음)
    Q_heater_W : float, optional
        히터 출력 [W], 기본값은 0.0
    Returns:
    --------
    np.ndarray
        다음 시간 단계의 노드 온도 배열 [K]
    '''
    def __init__(self, H, N, r0, x_shell, x_ins, k_shell, k_ins, h_w, h_o, C_d_mix):
        self.H = H; self.D = 2*r0; self.N = N
        self.A = np.pi * (self.D**2) / 4.0
        self.dz = H / N
        self.V = self.A * self.dz
        self.UA = calc_UA_tank_arr(r0, x_shell, x_ins, k_shell, k_ins, H, N, h_w, h_o)
        self.K = k_w * self.A / self.dz
        self.C = rho_w * c_w * self.V
        self.C_d_mix = C_d_mix
        
    # --- 추가: 유틸리티 헬퍼 (클래스 바깥에 둬도 됨) -----------------------------

        
    def step(self,
             T, dt, T_in, dV, T_amb,
             Q_heater_node=None, Q_heater_W=0.0,
             loop_outlet_node=None, loop_inlet_node=None,
             dV_loop=0.0, Q_loop=0.0):
        """
        기존 인자 + 외부 루프(탱크↔열교환기) 인자를 추가:
        - loop_outlet_node: 탱크에서 루프로 '빼내는' 노드 (1..N)
        - loop_inlet_node : 루프에서 탱크로 '되돌리는' 노드 (1..N)
        - dV_loop        : 루프 체적유량 [m^3/s]
        - Q_loop         : 루프 가열률(전열량, W)
        """
        N = self.N
        UA = self.UA; K = self.K
        G = c_w * rho_w * dV                            # 기존 전구간 대류
        eps = 1e-12
        G_loop = c_w * rho_w * max(dV_loop, 0.0)        # 루프 대류

        # ---- 부시네스크 혼합 (기존) ------------------------------------------------
        G_mix = np.zeros(N - 1)
        for i in range(N - 1):
            Q_mix = calc_boussinesq_mixing_flow(T[i], T[i+1], self.A, self.dz, self.C_d_mix)
            G_mix[i] = rho_w * c_w * Q_mix

        # ---- TDMA 계수 기본 구성 ----------------------------------------------------
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
        S = np.zeros(N)

        if Q_heater_node is not None:
            idx = Q_heater_node - 1
            if 0 <= idx < N:
                S[idx] = Q_heater_W

        # 상부(노드 1, idx=0) : 유출 경계 (전구간 G는 상향 가정일 때 top에 미적용)
        a[0] = 0.0
        b[0] = self.C/dt + K + UA[0] + G_mix[0]
        c[0] = -(K + G_mix[0])
        d[0] = self.C*T[0]/dt + UA[0]*T_amb + S[0]

        # 내부(i=2..N-1)
        for i in range(1, N-1):
            a[i]   = -(K + G_mix[i-1])
            b[i]   = self.C/dt + (2*K) + UA[i] + (G if dV>0 else 0.0) + G_mix[i-1] + G_mix[i]
            c[i]  = -(K + (G if dV>0 else 0.0) + G_mix[i])
            d[i]   = self.C*T[i]/dt + UA[i]*T_amb + S[i]

        # 하부(노드 N, idx=N-1) : 유입 경계
        a[N-1] = -(K + G_mix[N-2] + (G if dV>0 else 0.0))
        b[N-1] = self.C/dt + K + UA[N-1] + G_mix[N-2] + (G if dV>0 else 0.0)
        c[N-1] = 0.0
        d[N-1] = self.C*T[N-1]/dt + UA[N-1]*T_amb + S[N-1] + (G if dV>0 else 0.0)*T_in

        # ---- 외부 루프(지정 구간 강제 대류) 반영 ------------------------------------
        if (G_loop > 0.0) and (loop_outlet_node is not None) and (loop_inlet_node is not None):
            out_idx = int(loop_outlet_node) - 1
            in_idx  = int(loop_inlet_node)  - 1
            if 0 <= out_idx < N and 0 <= in_idx < N and out_idx != in_idx:
                # 루프 스트림 유입 온도 (outlet 측 온도 기준)
                T_stream_out = T[out_idx]                           # n 시점 사용(안정적)
                T_loop_in = T_stream_out + Q_loop / max(G_loop, eps)
                # (선택) 비현실적 고온 방지용 소프트 클램프 예시:
                # T_loop_in = min(T_loop_in, T_stream_out + 50.0)

                _add_loop_advection_terms(a, b, c, d, in_idx, out_idx, G_loop, T_loop_in)

        # ---- 선형계 풀이 ------------------------------------------------------------
        T_next = TDMA(a, b, c, d)
        return T_next

