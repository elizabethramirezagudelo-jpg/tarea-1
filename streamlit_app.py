import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="La Máquina Perceptrón - Welch Labs Style", layout="wide")

st.title("🕹️ La Máquina Perceptrón Interactiva")
st.markdown("""
Inspirado en el diseño físico de **Welch Labs**. Ajusta manualmente las perillas (pesos y bias) 
para intentar clasificar los patrones. ¡El aprendizaje automático eres TÚ!
""")

st.sidebar.header("Perillas (Pesos y Bias)")

# 1. Sliders para los Pesos (Manuales, sin entrenamiento automático)
w1 = st.sidebar.slider("Peso $w_1$ (Entrada X₁)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
w2 = st.sidebar.slider("Peso $w_2$ (Entrada X₂)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
bias = st.sidebar.slider("Bias (Sesgo $b$)", min_value=-5.0, max_value=5.0, value=-0.5, step=0.1)

# Función de activación Step (Escalón Heaviside)
def activation_function(z):
    return 1 if z >= 0 else 0

# 2. Configuración de la Tabla de Verdad / Patrones
st.subheader("Configuración de Patrones y Etiquetas Deseadas")
st.write("Define qué salida esperas para cada combinación de entradas:")

col1, col2, col3, col4 = st.columns(4)
combinations = [(0,0), (0,1), (1,0), (1,1)]
desired_outputs = {}

with col1:
    st.markdown("**Patrón 1 (0,0)**")
    y_00 = st.radio("Clase deseada (0,0):", [0, 1], index=0, key="y00")
with col2:
    st.markdown("**Patrón 2 (0,1)**")
    y_01 = st.radio("Clase deseada (0,1):", [0, 1], index=1, key="y01")
with col3:
    st.markdown("**Patrón 3 (1,0)**")
    y_10 = st.radio("Clase deseada (1,0):", [0, 1], index=1, key="y10")
with col4:
    st.markdown("**Patrón 4 (1,1)**")
    y_11 = st.radio("Clase deseada (1,1):", [0, 1], index=1, key="y11")

labels = [y_00, y_01, y_10, y_11]

# 3. Cálculos del Perceptrón
correct_count = 0
results_data = []

for (x1, x2), target in zip(combinations, labels):
    # Suma ponderada: z = w1*x1 + w2*x2 + bias
    z = (x1 * w1) + (x2 * w2) + bias
    output = activation_function(z)
    is_correct = (output == target)
    if is_correct:
        correct_count += 1
    
    results_data.append({
        "Inputs": f"({x1}, {x2})",
        "Target": target,
        "Suma (z)": round(z, 2),
        "Predicción": output,
        "Estado": "Correcto" if is_correct else "❌ Incorrecto"
    })

# Mostrar Score en vivo
st.markdown("---")
if correct_count == 4:
    st.success(f"¡Felicidades! Clasificaste correctamente todos los patrones ({correct_count}/4).")
else:
    st.warning(f"Patrones correctos: {correct_count} de 4. ¡Sigue ajustando las perillas!")

# Mostrar tabla de resultados interactiva
st.table(results_data)

# 4. Visualización de la Frontera de Decisión con Plotly
st.subheader("Frontera de Decisión en Tiempo Real")

fig = go.Figure()

# Dibujar la línea de decisión: w1*x1 + w2*x2 + b = 0  =>  x2 = (-w1*x1 - b) / w2
x1_range = np.linspace(-0.5, 1.5, 100)
if w2 != 0:
    x2_range = (-w1 * x1_range - bias) / w2
    # Filtrar valores para evitar que el gráfico se deforme horriblemente si w2 se acerca a 0
    valid = (x2_range >= -1) & (x2_range <= 2)
    fig.add_trace(go.Scatter(
        x=x1_range[valid], y=x2_range[valid],
        mode='lines', name='Frontera ($z=0$)',
        line=dict(color='firebrick', width=3, dash='dash')
    ))
else:
    # Si w2 es cero, la línea es vertical: x1 = -b / w1
    if w1 != 0:
        x1_vert = -bias / w1
        fig.add_vline(x=x1_vert, line_width=3, line_dash="dash", line_color="firebrick", name="Frontera ($z=0$)")

# Dibujar los 4 puntos (patrones)
for (x1, x2), target in zip(combinations, labels):
    color = 'blue' if target == 1 else 'orange'
    symbol = 'circle' if target == 1 else 'x'
    fig.add_trace(go.Scatter(
        x=[x1], y=[x2],
        mode='markers+text',
        marker=dict(color=color, size=15, symbol=symbol, line=dict(width=2, color='black')),
        text=[f"({x1},{x2}) Target: {target}"],
        textposition="top center",
        name=f"Clase {target}"
    ))

# Configuración estética del plano cartesiano
fig.update_layout(
    xaxis=dict(range=[-0.5, 1.5], title="Entrada X₁"),
    yaxis=dict(range=[-0.5, 1.5], title="Entrada X₂"),
    width=700, height=500,
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
