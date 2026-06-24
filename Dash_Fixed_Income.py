"""
\item \textbf{Dashboard de Renta Fija}: Desarrollé una plataforma en Python para la valuación y gestión del riesgo de mercado de deuda soberana mexicana (M-Bonos, CETES, UDIBonos). Programé motores financieros para el cálculo bidireccional Precio/YTM, segregación de precio limpio/sucio e ingesta automatizada de variables macroeconómicas mediante APIs. Modelé la sensibilidad del portafolio ante movimientos en la curva de tasas a través de métricas core: Duración Modificada, Convexidad y DV01. Asimismo, implementé un módulo de \textit{stress testing} para simular impactos inmediatos ante shocks en la estructura temporal ($\pm1$ a $\pm100$ bps) y diseñé visualizaciones interactivas para analizar efectos no lineales en la valuación.
\item \textbf{Dashboard de Renta Fija (Python & Streamlit)}: Diseñé y desarrollé una plataforma analítica modular para la valuación y gestión de riesgo de mercado de deuda soberana en México (M-Bonos, CETES, UDIBonos). Programé motores financieros para el cálculo bidireccional de Precio/YTM, segregación de precio limpio/sucio y automatización de indicadores macroeconómicos (UDIs) mediante conexiones seguras a APIs. Implementé algoritmos cuantitativos para modelar la sensibilidad de la curva a través de métricas avanzadas: Duración Modificada, Convexidad y DV01. Estructuré un módulo de \textit{Stress Testing} para simular impactos inmediatos ante shocks en la estructura temporal de tasas ($\pm1$ a $\pm100$ bps) y desarrollé visualizaciones interactivas para capturar efectos no lineales en los portafolios.

"""
import numpy as np
import pandas as pd
import requests
import scipy.optimize as optimize

# ==========================================
# 1. CONEXIÓN CON API (BANXICO)
# ==========================================

def obtener_udi_actual(api_key: str, date: str) -> float:
    """Obtiene el valor de la UDI desde la API de Banxico."""
    headers = {'Bmx-Token': api_key}
    udi_series_id = "SP68257"
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{udi_series_id}/datos/{date}/{date}"

    try:
        respuesta = requests.get(url, headers=headers, timeout=10)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            if 'bmx' in datos and 'series' in datos['bmx'] and len(datos['bmx']['series'][0]['datos']) > 0:
                return float(datos['bmx']['series'][0]['datos'][0]['dato'])
            raise ValueError(f"No se encontraron datos de UDI para la fecha: {date}")
        raise Exception(f"Error de API Banxico (Status: {respuesta.status_code})")
    except Exception as e:
        raise Exception(f"Error al conectar con Banxico: {str(e)}")


# ==========================================
# 2. MOTOR DE VALUACIÓN, RIESGO Y OPTIMIZACIÓN (M-BONOS)
# ==========================================

def calcular_ytm_mbono(precio_sucio: float, tasa_cupon: float, dias_vencer: int, valor_nominal: float = 100.0, dias_cupon: int = 182, guess: float = 0.05) -> float:
    """
    Calcula de forma exacta el Yield to Maturity (YTM) a partir del precio sucio 
    utilizando el método numérico de Newton-Raphson (Rescatado de tu código).
    """
    c_anual = tasa_cupon / 100
    pago_cupon = valor_nominal * ((c_anual * dias_cupon) / 360)
    periodos_restantes = int(np.ceil(dias_vencer / dias_cupon))
    t_fraccion = (dias_vencer % dias_cupon) / dias_cupon if dias_vencer % dias_cupon != 0 else 1.0
    
    tiempos_cupon = np.array([t_fraccion + i for i in range(periodos_restantes)])
    
    flujos = np.zeros(periodos_restantes)
    flujos[:-1] = pago_cupon
    flujos[-1] = pago_cupon + valor_nominal

    # Función objetivo para Newton-Raphson: Valor Presente(y) - Precio = 0
    def ytm_objetivo(y):
        y_periodo = (y * dias_cupon) / 360
        factores_descuento = 1 / (1 + y_periodo)**tiempos_cupon
        return np.sum(flujos * factores_descuento) - precio_sucio

    try:
        ytm_calculado = optimize.newton(ytm_objetivo, guess, tol=1e-6, maxiter=200)
        return float(ytm_calculado * 100)  # Retorna en formato porcentaje (ej. 8.5)
    except RuntimeError:
        # Fallback por si no converge Newton: buscar en un rango seguro
        try:
            return float(optimize.brentq(ytm_objetivo, -0.2, 2.0) * 100)
        except Exception:
            raise ValueError("No se pudo calcular el YTM para el precio e instrumento dados.")


def evaluar_mbono(tasa_cupon: float, dias_vencer: int, tasa_rendimiento: float, valor_nominal: float = 100.0, dias_cupon: int = 182) -> dict:
    """
    Calcula valuación analítica completa, flujos descontados, sensibilidades analíticas,
    sensibilidades por diferencias finitas (método numérico dy) y análisis de Shocks.
    """
    c_anual = tasa_cupon / 100
    y_anual = tasa_rendimiento / 100
    frecuencia = 360 / dias_cupon
    
    pago_cupon = valor_nominal * ((c_anual * dias_cupon) / 360)
    periodos_restantes = int(np.ceil(dias_vencer / dias_cupon))
    dias_transcurridos = dias_cupon - (dias_vencer % dias_cupon) if dias_vencer % dias_cupon != 0 else 0
    
    y_periodo = (y_anual * dias_cupon) / 360
    t_fraccion = (dias_vencer % dias_cupon) / dias_cupon if dias_vencer % dias_cupon != 0 else 1.0

    # ---- CONSTRUCCIÓN DE VECTOR DE FLUJOS ----
    flujos = np.zeros(periodos_restantes)
    flujos[:-1] = pago_cupon
    flujos[-1] = pago_cupon + valor_nominal
    tiempos_cupon = np.array([t_fraccion + i for i in range(periodos_restantes)])
    
    factores_descuento = 1 / (1 + y_periodo)**tiempos_cupon
    valores_presentes = flujos * factores_descuento
    
    precio_sucio = np.sum(valores_presentes)
    acumulado_cupon = valor_nominal * ((dias_transcurridos * c_anual) / 360)
    precio_limpio = precio_sucio - acumulado_cupon

    # ---- MÉTRICAS DE RIESGO (FÓRMULAS ANALÍTICAS) ----
    duracion_macaulay_anos = np.sum(tiempos_cupon * valores_presentes) / (precio_sucio * frecuencia)
    duracion_modificada_anual = duracion_macaulay_anos / (1 + y_periodo)
    
    terminos_convexidad = tiempos_cupon * (tiempos_cupon + 1) * valores_presentes
    convexidad_anual = (1 / precio_sucio) * (1 / (1 + y_periodo)**2) * np.sum(terminos_convexidad) * (1 / frecuencia**2)
    dv01 = precio_sucio * duracion_modificada_anual * 0.0001

    # ---- SENSIVILIDAD NUMÉRICA POR DIFERENCIAS FINITAS (Rescatado de tu código) ----
    dy = 0.01  # Shock de 1% para la aproximación numérica
    
    def calcular_precio_aux(y_shock):
        y_p = (y_shock * dias_cupon) / 360
        fd = 1 / (1 + y_p)**tiempos_cupon
        return np.sum(flujos * fd)
        
    precio_minus = calcular_precio_aux(y_anual - dy)
    precio_plus = calcular_precio_aux(y_anual + dy)
    
    duracion_modificada_numerica = (precio_minus - precio_plus) / (2 * precio_sucio * dy)
    convexidad_numerica = (precio_minus + precio_plus - 2 * precio_sucio) / (precio_sucio * (dy ** 2))

    # ---- DATAFRAME DE FLUJOS PARA TABLAS EN EL DASHBOARD ----
    df_flujos = pd.DataFrame({
        "Periodo": np.arange(1, periodos_restantes + 1),
        "Tiempo_Anos": tiempos_cupon / frecuencia,
        "Flujo_Efectivo": flujos,
        "Valor_Presente": valores_presentes
    })

    # ---- SIMULACIÓN DE SHOCKS (Aproximación de Taylor) ----
    shocks = [-0.01, -0.0025, -0.0001, 0.0001, 0.0025, 0.01]
    resultados_shocks = {}
    
    for s in shocks:
        efecto_duracion = -duracion_modificada_anual * s
        efecto_convexidad = 0.5 * convexidad_anual * (s**2)
        efecto_total = efecto_duracion + efecto_convexidad
        
        cambio_precio_estimado = precio_sucio * efecto_total
        
        resultados_shocks[f"shock_{int(s*10000)}bps"] = {
            "cambio_tasa_bps": int(s * 10000),
            "efecto_duracion": efecto_duracion,
            "efecto_convexidad": efecto_convexidad,
            "cambio_porcentual_total": efecto_total,
            "cambio_precio_mxn": cambio_precio_estimado,
            "precio_estimado_sucio": precio_sucio + cambio_precio_estimado
        }

    return {
        "precio_limpio": precio_limpio,
        "precio_sucio": precio_sucio,
        "interes_devengado": acumulado_cupon,
        "duracion_macaulay": duracion_macaulay_anos,
        "duracion_modificada": duracion_modificada_anual,
        "duracion_modificada_num": duracion_modificada_numerica,  # Nuevo analítico comparativo
        "convexidad": convexidad_anual,
        "convexidad_num": convexidad_numerica,                    # Nuevo analítico comparativo
        "dv01": dv01,
        "tabla_flujos": df_flujos,
        "analisis_shocks": resultados_shocks
    }


# ==========================================
# 3. MOTOR DE VALUACIÓN (CETES)
# ==========================================

def evaluar_cetes(dias_vencer: int, tasa: float, es_descuento: bool, valor_nominal: float = 10.0) -> dict:
    """
    Calcula valuación de CETES incorporando la lógica de Descuento comercial 
    vs. Rendimiento (Interés Simple / Zero Coupon Bond).
    """
    tasa_dec = tasa / 100
    tiempo_anos = dias_vencer / 360
    
    if es_descuento:
        # Valuación basada en Descuento Puro
        precio = valor_nominal * (1 - (tasa_dec * tiempo_anos))
        tasa_descuento = tasa_dec
        tasa_rendimiento = ((valor_nominal / precio) - 1) / tiempo_anos
    else:
        # Valuación basada en Rendimiento (ZCB / Interés Simple)
        precio = valor_nominal / (1 + (tasa_dec * tiempo_anos))
        tasa_rendimiento = tasa_dec
        tasa_descuento = tasa_rendimiento / (1 + tasa_rendimiento * tiempo_anos)

    # Duración para un bono cupón cero es igual a su tiempo al vencimiento
    duracion_modificada = tiempo_anos / (1 + tasa_rendimiento * tiempo_anos)
    dv01 = precio * duracion_modificada * 0.0001

    return {
        "precio": precio,
        "tasa_descuento_anual": tasa_descuento * 100,
        "tasa_rendimiento_anual": tasa_rendimiento * 100,
        "duracion_anos": tiempo_anos,
        "dv01": dv01
    }


# ==========================================
# 4. MOTOR DE VALUACIÓN Y SENSIBILIDAD (UDIBONOS)
# ==========================================

def evaluar_udibonos(tasa_cupon: float, dias_vencer: int, tasa_rendimiento_real: float, api_key: str, date: str, valor_nominal_udis: float = 100.0, dias_cupon: int = 182) -> dict:
    """Valúa un UDIBono acoplando la API de la UDI con las sensibilidades analíticas y numéricas."""
    udi_val = obtener_udi_actual(api_key, date)
    
    analisis_en_udis = evaluar_mbono(
        tasa_cupon=tasa_cupon,
        dias_vencer=dias_vencer,
        tasa_rendimiento=tasa_rendimiento_real,
        valor_nominal=valor_nominal_udis,
        dias_cupon=dias_cupon
    )
    
    df_flujos_mxn = analisis_en_udis["tabla_flujos"].copy()
    df_flujos_mxn["Flujo_Efectivo_MXN"] = df_flujos_mxn["Flujo_Efectivo"] * udi_val
    df_flujos_mxn["Valor_Presente_MXN"] = df_flujos_mxn["Valor_Presente"] * udi_val

    shocks_mxn = analisis_en_udis["analisis_shocks"].copy()
    for k in shocks_mxn.keys():
        shocks_mxn[k]["cambio_precio_mxn"] = shocks_mxn[k]["cambio_precio_mxn"] * udi_val
        shocks_mxn[k]["precio_estimado_sucio_mxn"] = shocks_mxn[k]["precio_estimado_sucio"] * udi_val

    return {
        "udi_valor_utilizado": udi_val,
        "precio_limpio_udi": analisis_en_udis["precio_limpio"],
        "precio_sucio_udi": analisis_en_udis["precio_sucio"],
        "precio_limpio_mxn": analisis_en_udis["precio_limpio"] * udi_val,
        "precio_sucio_mxn": analisis_en_udis["precio_sucio"] * udi_val,
        "interes_devengado_mxn": analisis_en_udis["interes_devengado"] * udi_val,
        "duracion_macaulay": analisis_en_udis["duracion_macaulay"],
        "duracion_modificada": analisis_en_udis["duracion_modificada"],
        "duracion_modificada_num": analisis_en_udis["duracion_modificada_num"],
        "convexidad": analisis_en_udis["convexidad"],
        "convexidad_num": analisis_en_udis["convexidad_num"],
        "dv01_mxn": analisis_en_udis["dv01"] * udi_val,
        "tabla_flujos": df_flujos_mxn,
        "analisis_shocks": shocks_mxn
    }