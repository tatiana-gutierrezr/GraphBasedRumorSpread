import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
import numpy as np
from scipy.integrate import solve_ivp

#Configuración de la app
st.set_page_config(page_title="Simulación de Rumores", layout="wide")
st.title("🌍 Propagación de Rumores en Redes Sociales")

#Parámetros
st.sidebar.header("🔧 Configuración de la Red")
n_nodes = st.sidebar.slider("Número de usuarios", 10, 200, 50)
graph_type = st.sidebar.selectbox("Tipo de red", ["Comunitaria", "Watts-Strogatz", "Barabasi-Albert", "Erdos-Renyi"])

#Parámetros específicos por tipo de red
if graph_type == "Watts-Strogatz":
    k = st.sidebar.slider("Vecinos por nodo (k)", 2, 10, 4)
    p = st.sidebar.slider("Prob. de reconexión (p)", 0.01, 1.0, 0.3)
elif graph_type == "Barabasi-Albert":
    m = st.sidebar.slider("Conexiones nuevas (m)", 1, 5, 2)
elif graph_type == "Erdos-Renyi":
    p = st.sidebar.slider("Prob. de conexión", 0.01, 0.5, 0.15)
else:  #Red comunitaria
    n_comunidades = st.sidebar.slider("Número de comunidades", 2, 10, 4)
    prob_inter = st.sidebar.slider("Prob. conexión inter-comunidad", 0.01, 0.3, 0.05)
    prob_intra = st.sidebar.slider("Prob. conexión intra-comunidad", 0.1, 1.0, 0.8)

#Parámetros de comportamiento
st.sidebar.header("🧠 Comportamientos Sociales")
beta = st.sidebar.slider("Prob. de infección (beta)", 0.01, 1.0, 0.7)
alpha = st.sidebar.slider("Prob. de recuperación (alpha)", 0.0, 0.5, 0.1)
immune_perc = st.sidebar.slider("% Usuarios inmunes", 0, 30, 8)
always_spread_perc = st.sidebar.slider("% Usuarios que siempre propagan", 0, 30, 5)
never_recover_perc = st.sidebar.slider("% Usuarios que nunca se recuperan", 0, 30, 12)
skeptic_perc = st.sidebar.slider("% Usuarios escépticos (difícil infectar)", 0, 40, 15)
forgetful_perc = st.sidebar.slider("% Usuarios olvidadizos (recuperación rápida)", 0, 30, 10)

#Configuración de simulación
steps = st.sidebar.slider("Pasos de tiempo", 10, 200, 80)
speed = st.sidebar.slider("Velocidad de animación", 0.01, 1.0, 0.2)
random_seed = st.sidebar.number_input("Semilla aleatoria", 0, 1000, 42)

#Creación del grafo con comportamientos realistas (como las redes sociales)
def create_realistic_graph(n, graph_type):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if graph_type == "Watts-Strogatz":
        G = nx.watts_strogatz_graph(n=n, k=k, p=p)
    elif graph_type == "Barabasi-Albert":
        G = nx.barabasi_albert_graph(n=n, m=m)
    elif graph_type == "Erdos-Renyi":
        G = nx.erdos_renyi_graph(n=n, p=p)
    else:
        G = nx.Graph()
        tamaño_comunidad = n // n_comunidades
        
        #Crear comunidades con conexiones probabilísticas
        for i in range(n_comunidades):
            comunidad = list(range(i*tamaño_comunidad, (i+1)*tamaño_comunidad if i != n_comunidades-1 else n))
            for u in comunidad:
                for v in comunidad:
                    if u != v and random.random() < prob_intra:
                        G.add_edge(u, v)
        
        #Conexiones entre comunidades
        for i in range(n_comunidades):
            comunidad_actual = list(range(i*tamaño_comunidad, (i+1)*tamaño_comunidad if i != n_comunidades-1 else n))
            for j in range(i+1, n_comunidades):
                comunidad_vecina = list(range(j*tamaño_comunidad, (j+1)*tamaño_comunidad if j != n_comunidades-1 else n))
                for u in comunidad_actual:
                    for v in comunidad_vecina:
                        if random.random() < prob_inter:
                            G.add_edge(u, v)
    
    #Asegurar que el grafo esté conectado
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        for component in components:
            if component != largest:
                u = random.choice(list(component))
                v = random.choice(list(largest))
                G.add_edge(u, v)
    
    #Asignar características a los usuarios
    for node in G.nodes:
        G.nodes[node]['state'] = 'I'  #Todos comienzan como ignorantes
        G.nodes[node]['threshold'] = random.uniform(0.05, 0.3)
        G.nodes[node]['immune'] = random.random() < (immune_perc/100)
        G.nodes[node]['always_spread'] = random.random() < (always_spread_perc/100)
        G.nodes[node]['never_recover'] = random.random() < (never_recover_perc/100)
        G.nodes[node]['skeptic'] = random.random() < (skeptic_perc/100)
        G.nodes[node]['forgetful'] = random.random() < (forgetful_perc/100)
        G.nodes[node]['influence'] = random.uniform(0.5, 3.0)
        
        #Ajustar características basadas en tipos
        if G.nodes[node]['skeptic']:
            G.nodes[node]['threshold'] = random.uniform(0.4, 0.8)
        if G.nodes[node]['forgetful']:
            G.nodes[node]['memory'] = random.uniform(0.1, 0.5)  #Memoria más corta
    
    #Seleccionar nodos iniciales (no inmunes) para asegurar propagación
    non_immune = [n for n in G.nodes if not G.nodes[n]['immune']]
    if non_immune:
        for start_node in random.sample(non_immune, min(5, len(non_immune))):
            G.nodes[start_node]['state'] = 'S'
            G.nodes[start_node]['initial_spreader'] = True
    
    return G

def realistic_simulation_step(G, beta, alpha):
    changes = {}
    
    for node in G.nodes:
        current_state = G.nodes[node]['state']
        
        #1. Procesar recuperaciones (S -> R)
        if current_state == 'S':
            #Usuarios olvidadizos se recuperan más rápido
            recovery_prob = alpha * (3.0 if G.nodes[node]['forgetful'] else 1.0)
            
            if not G.nodes[node]['always_spread'] and not G.nodes[node]['never_recover']:
                if random.random() < recovery_prob:
                    changes[node] = 'R'
        
        #2. Procesar nuevas infecciones (I -> S)
        elif current_state == 'I' and not G.nodes[node]['immune']:
            neighbors = list(G.neighbors(node))
            if neighbors:
                spreaders = [n for n in neighbors if G.nodes[n]['state'] == 'S']
                if spreaders:
                    #Escépticos son más difíciles de infectar
                    infection_factor = 0.3 if G.nodes[node]['skeptic'] else 1.0
                    #Influencia de los propagadores
                    infection_pressure = sum(G.nodes[n]['influence'] for n in spreaders) / len(neighbors)
                    
                    if infection_pressure > G.nodes[node]['threshold']:
                        adjusted_beta = beta * (infection_pressure - G.nodes[node]['threshold']) * infection_factor
                        if random.random() < adjusted_beta:
                            changes[node] = 'S'
    
    #Aplicar cambios
    for node, new_state in changes.items():
        G.nodes[node]['state'] = new_state
    
    return G

#Modelo ODE para comparación
def rumor_odes(t, y, beta, alpha, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - alpha * I
    dRdt = alpha * I
    return [dSdt, dIdt, dRdt]

#Crear grafo y posiciones fijas
G = create_realistic_graph(n_nodes, graph_type)
pos = nx.spring_layout(G, seed=random_seed)

#Configuración de colores para cada nodo
color_map = {
    'I': '#CCCCCC',  #Ignorantes - Gris claro
    'S': '#FFA500',  #Propagadores normales - Naranja
    'R': '#4682B4',  #Recuperados - Azul acero
    'immune': '#333333',  #Inmunes - Gris oscuro
    'always_spread': '#FF0000',  #Siempre propagan - Rojo
    'never_recover': '#8B0000',  #Nunca se recuperan - Rojo oscuro
    'skeptic': '#2E8B57',  #Escépticos - Verde mar
    'forgetful': '#9370DB',  #Olvidadizos - Lila
    'initial': '#FFD700'  #Propagadores iniciales - Amarillo
}

#Tabs principales
tab1, tab2 = st.tabs(["🏃 Simulación", "📚 Análisis Matemático"])

with tab1:
    #Simulación principal
    st.subheader("🔄 Dinámica de Propagación")
    animation_placeholder = st.empty()
    state_counts = []
    history = []

    #Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(steps):
        G = realistic_simulation_step(G, beta, alpha)
        
        #Contar estados
        counts = {
            'I': sum(1 for n in G.nodes if G.nodes[n]['state'] == 'I' and not G.nodes[n]['immune']),
            'S_normal': sum(1 for n in G.nodes if G.nodes[n]['state'] == 'S' and not G.nodes[n]['never_recover'] and not G.nodes[n]['always_spread']),
            'S_permanent': sum(1 for n in G.nodes if G.nodes[n]['state'] == 'S' and G.nodes[n]['never_recover']),
            'S_always': sum(1 for n in G.nodes if G.nodes[n]['state'] == 'S' and G.nodes[n]['always_spread']),
            'R': sum(1 for n in G.nodes if G.nodes[n]['state'] == 'R'),
            'Immune': sum(1 for n in G.nodes if G.nodes[n]['immune']),
            'Skeptic': sum(1 for n in G.nodes if G.nodes[n]['skeptic'] and G.nodes[n]['state'] in ['I', 'S']),
            'Forgetful': sum(1 for n in G.nodes if G.nodes[n]['forgetful'] and G.nodes[n]['state'] == 'R')
        }
        state_counts.append(counts)
        history.append([G.nodes[n]['state'] for n in G.nodes])
        
        #Actualizar progreso
        progress = (step + 1) / steps
        progress_bar.progress(progress)
        status_text.text(f"Progreso: {step+1}/{steps} pasos | Propagadores: {counts['S_normal'] + counts['S_permanent'] + counts['S_always']}")
        
        #Visualización del grafo
        fig, ax = plt.subplots(figsize=(12, 9))
        
        node_colors = []
        node_sizes = []
        for n in G.nodes:
            if G.nodes[n]['immune']:
                node_colors.append(color_map['immune'])
                node_sizes.append(40)
            elif 'initial_spreader' in G.nodes[n] and G.nodes[n]['initial_spreader'] and G.nodes[n]['state'] == 'S':
                node_colors.append(color_map['initial'])
                node_sizes.append(120)
            elif G.nodes[n]['always_spread'] and G.nodes[n]['state'] == 'S':
                node_colors.append(color_map['always_spread'])
                node_sizes.append(100)
            elif G.nodes[n]['never_recover'] and G.nodes[n]['state'] == 'S':
                node_colors.append(color_map['never_recover'])
                node_sizes.append(90)
            elif G.nodes[n]['skeptic']:
                node_colors.append(color_map['skeptic'] if G.nodes[n]['state'] == 'I' else color_map[G.nodes[n]['state']])
                node_sizes.append(80)
            elif G.nodes[n]['forgetful']:
                node_colors.append(color_map['forgetful'] if G.nodes[n]['state'] == 'R' else color_map[G.nodes[n]['state']])
                node_sizes.append(70)
            else:
                node_colors.append(color_map[G.nodes[n]['state']])
                node_sizes.append(60)
        
        nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, 
                with_labels=False, edge_color='lightgray', ax=ax, width=0.8, alpha=0.8)
        
        #Leyenda personalizada para cada paso
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Ignorantes', markerfacecolor=color_map['I'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Propagadores normales', markerfacecolor=color_map['S'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Propagadores permanentes', markerfacecolor=color_map['never_recover'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Siempre propagan', markerfacecolor=color_map['always_spread'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Recuperados', markerfacecolor=color_map['R'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Inmunes', markerfacecolor=color_map['immune'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Escépticos', markerfacecolor=color_map['skeptic'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Olvidadizos', markerfacecolor=color_map['forgetful'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Iniciales', markerfacecolor=color_map['initial'], markersize=10)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        ax.set_title(f"Paso {step+1}/{steps}\nIgnorantes={counts['I']} | Propagadores={counts['S_normal']} | " +
                    f"Permanentes={counts['S_permanent']} | Siempre={counts['S_always']} | Recuperados={counts['R']}",
                    fontsize=12)
        
        animation_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(speed)

    #Resultados finales
    st.subheader("📊 Resultados Finales")
    final_counts = state_counts[-1]
    total_S = final_counts['S_normal'] + final_counts['S_permanent'] + final_counts['S_always']
    
    st.write(f"""
    - **Ignorantes (nunca creyeron):** {final_counts['I']} ({final_counts['I']/n_nodes*100:.1f}%)
    - **Propagadores activos:** {final_counts['S_normal']} ({final_counts['S_normal']/n_nodes*100:.1f}%)
    - **Propagadores permanentes:** {final_counts['S_permanent']} ({final_counts['S_permanent']/n_nodes*100:.1f}%)
    - **Usuarios que siempre propagan:** {final_counts['S_always']} ({final_counts['S_always']/n_nodes*100:.1f}%)
    - **Recuperados:** {final_counts['R']} ({final_counts['R']/n_nodes*100:.1f}%)
    - **Usuarios inmunes:** {final_counts['Immune']} ({final_counts['Immune']/n_nodes*100:.1f}%)
    - **Usuarios escépticos (no convencidos o propagadores):** {final_counts['Skeptic']} ({final_counts['Skeptic']/n_nodes*100:.1f}%)
    - **Usuarios olvidadizos (ya recuperados):** {final_counts['Forgetful']} ({final_counts['Forgetful']/n_nodes*100:.1f}%)
    """)

    #Gráfico de evolución temporal
    st.subheader("📈 Evolución del Rumor")
    evolution_data = pd.DataFrame(state_counts)
    evolution_data['Total_S'] = evolution_data['S_normal'] + evolution_data['S_permanent'] + evolution_data['S_always']
    st.line_chart(evolution_data[['I', 'Total_S', 'R']])

with tab2:
    st.subheader("📐 Teoría de Sistemas Dinámicos Aplicada")
    
    with st.expander("🔍 Análisis Cualitativo", expanded=True):
        st.markdown("""
        **1. Linealización y Estabilidad:**
        - Determinante:
        """)
        st.latex(r'''
        J = \begin{bmatrix}
        0 & -\beta \\
        0 & \beta - \alpha
        \end{bmatrix}
        ''')
        st.write(f"Eigenvalores: λ₁ = {0}, λ₂ = {beta - alpha:.2f}")
        
        if beta > alpha:
            st.error("λ₂ > 0 ⇒ Punto inestable (rumor se propaga)")
        else:
            st.success("λ₂ < 0 ⇒ Punto estable (rumor desaparece)")
        
        st.markdown("""
        **2. Bifurcación:**  
        Cambio cualitativo cuando $R_0 = 1$ ($\\beta = \\alpha/N$).
        """)
        
        current_S = sum(1 for n in G.nodes if G.nodes[n]['state'] == 'I')
        current_I = sum(1 for n in G.nodes if G.nodes[n]['state'] == 'S')

        if current_I > 0 and current_S > 0:
            J_current = np.array([
                [-beta * current_I/n_nodes, -beta * current_S/n_nodes],
                [beta * current_I/n_nodes, beta * current_S/n_nodes - alpha]
            ])
            eigenvals_current = np.linalg.eigvals(J_current)
            
            #Formateo mejorado con LaTeX
            st.markdown("*Matriz Jacobiana actual:*")
            st.latex(r"""
            J_{\text{actual}} = 
            \begin{bmatrix} 
            -\beta \cdot \frac{I}{N} & -\beta \cdot \frac{S}{N} \\ 
            \beta \cdot \frac{I}{N} & \beta \cdot \frac{S}{N} - \alpha 
            \end{bmatrix} =
            \begin{bmatrix} 
            %.4f & %.4f \\ 
            %.4f & %.4f 
            \end{bmatrix}
            """ % (
                -beta * current_I/n_nodes, 
                -beta * current_S/n_nodes,
                beta * current_I/n_nodes, 
                beta * current_S/n_nodes - alpha
            ))
            
            st.markdown("*Eigenvalores (raíces características):*")
            
            #Mostrar ecuación general
            st.latex(r"""
            \lambda_{1,2} = \frac{\text{tr}(J) \pm \sqrt{\text{tr}(J)^2 - 4 \cdot \text{det}(J)}}{2}
            """)
            
            #Calcular componentes
            tr = np.trace(J_current)
            det = np.linalg.det(J_current)
            discriminant = tr**2 - 4*det
            
            #Mostrar valores exactos con notación científica si son muy pequeños
            def format_val(x):
                return "{:.6f}".format(x) if abs(x) > 1e-4 else "{:.4e}".format(x)
            
            st.latex(r"""
            \begin{aligned}
            \text{tr}(J) &= %.6f \\
            \text{det}(J) &= %.6f \\
            \Delta &= %.6f \\
            \lambda_1 &= %.6f \\
            \lambda_2 &= %.6f
            \end{aligned}
            """ % (tr, det, discriminant, eigenvals_current[0], eigenvals_current[1]))
            
            #Análisis de estabilidad
            st.markdown("*Interpretación:*")
            if all(np.real(e) < 0 for e in eigenvals_current):
                st.success("Sistema estable (ambos eigenvalores tienen parte real negativa)")
            elif any(np.real(e) > 0 for e in eigenvals_current):
                st.error("Sistema inestable (al menos un eigenvalor con parte real positiva)")
            else:
                st.warning("Sistema marginalmente estable (parte real cero)")
        
    with st.expander("📈 Retrato de Fase", expanded=True):
        #Configurar el espacio de fase
        S_vals = np.linspace(0, n_nodes, 20)
        I_vals = np.linspace(0, n_nodes, 20)
        S, I = np.meshgrid(S_vals, I_vals)
        
        #Ajustar ecuaciones con parámetros REALES de tu simulación
        beta_eff = beta * (1 - skeptic_perc/100)  #Efecto de escépticos
        alpha_eff = alpha * (1 + forgetful_perc/100)  #Efecto de olvidadizos
        
        dSdt = -beta_eff * S * I / n_nodes
        dIdt = beta_eff * S * I / n_nodes - alpha_eff * I
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        #Dibujar el campo vectorial
        ax.streamplot(S, I, dSdt, dIdt, density=1.5, color='lightgray')
        
        #Superponer la trayectoria REAL de la simulación
        S_sim = [step['I'] for step in state_counts]  #Ignorantes = Susceptibles
        I_sim = [step['S_normal'] + step['S_permanent'] + step['S_always'] for step in state_counts]  #Propagadores = Infectados
        ax.plot(S_sim, I_sim, 'r-', linewidth=2, label='Trayectoria real')
        
        #Marcar el estado INICIAL y FINAL
        ax.plot(S_sim[0], I_sim[0], 'go', markersize=8, label='Inicio')
        ax.plot(S_sim[-1], I_sim[-1], 'rx', markersize=10, label='Final')
        
        ax.set_xlabel('Susceptibles (Ignorantes)')
        ax.set_ylabel('Infectados (Propagadores)')
        ax.set_title(f'Retrato de Fase (β_eff={beta_eff:.2f}, α_eff={alpha_eff:.2f})')
        ax.legend()
        st.pyplot(fig)
        
        
    st.subheader("🔍 Análisis matemático avanzado")
    
    st.markdown("---")
    with st.expander("🔍 Modelo Matemático Completo", expanded=True):
        st.markdown("""
        ### 📜 Ecuaciones Fundamentales
        
        **Modelo SIR básico**:
        ```
        dS/dt = -β(SI/N)         #Susceptibles que se infectan
        dI/dt = β(SI/N) - αI     #Infectados que propagan/se recuperan
        dR/dt = αI               #Recuperados
        ```
        
        **Variables clave**:
        - S, I, R: Población en cada estado
        - N: Población total (S + I + R)
        - β: Tasa de contagio (slider)
        - α: Tasa de recuperación (slider)
        
        ---
        
        ### 🛠️ Extensiones del Modelo
        
        | Variable          | Símbolo | Efecto Matemático            | Ejemplo Código             |
        |-------------------|---------|-------------------------------|----------------------------|
        | Inmunes           | 🛡️      | Bloquean infección            | `node['immune'] = True`    |
        | Escépticos        | 🤔      | Reducen β en 70%              | `β *= 0.3 if skeptic`      |
        | Olvidadizos       | 🧠      | Triplican tasa recuperación   | `α *= 3.0 if forgetful`    |
        | Propag. Permanentes | 🔄   | No se recuperan nunca         | `if never_recover: ...`    |
        
        ---""")

    
    #Sección explicativa
    with st.expander("📚 Modelo SIR mejorado con comportamientos reales", expanded=True):
        st.markdown("""
        **1. Extensiones al Modelo Clásico:**
        
        Hemos extendido el modelo SIR tradicional para incluir:
        
        - **Propagadores permanentes (Sₚ):** Usuarios que nunca se recuperan (Sₚ → Sₚ)
        - **Usuarios escépticos:** Mayor umbral de infección (βₑ = f·β, donde f < 1)
        - **Usuarios olvidadizos:** Mayor tasa de recuperación (αₒ = 3·α)
        - **Inmunes:** No pueden infectarse (Iᵢ → Iᵢ)
        
        **2. Ecuaciones Modificadas:**
        
        Para cada nodo i en la red:
        """)
        st.latex(r'''
        \begin{cases}
        \frac{dI_i}{dt} = -\sum_j \beta_{ij} A_{ij} I_i S_j \\
        \frac{dS_i}{dt} = \sum_j \beta_{ij} A_{ij} I_i S_j - \alpha_i S_i \\
        \frac{dR_i}{dt} = \alpha_i S_i
        \end{cases}
        ''')
        st.markdown("""
        Donde:
        - $β_{ij} = β·(1 - f_j)$ (f_j = factor de escepticismo)
        - $α_i = α·m_i$ (m_i = factor de memoria, m_i > 1 para olvidadizos)
        - Sₚ no aparecen en las ecuaciones (son un estado absorbente)
        """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Parámetros Clave")
        N_eff = n_nodes - final_counts['Immune']
        total_S = final_counts['S_normal'] + final_counts['S_permanent'] + final_counts['S_always']
        
        st.write("**Población efectiva (N):**", N_eff)
        st.write("**Tasa de infección efectiva (β):**", beta)
        st.write("**Tasa de recuperación efectiva (α):**", alpha)
        
        if alpha > 0:
            R0_classic = beta * N_eff / alpha
            st.write(f"**Número reproductivo básico (R₀):** {R0_classic:.2f}")
            
            st.write("**Punto endémico teórico:**")
            I_star = alpha/beta if beta > 0 else 0
            S_star = N_eff - I_star if beta > 0 else N_eff
            st.latex(fr"""
            \begin{{aligned}}
            S^* &= {S_star:.1f} \\
            I^* &= {I_star:.1f} \\
            R^* &= 0
            \end{{aligned}}
            """)
        else:
            st.warning("Cuando α=0, no hay punto endémico estable (el rumor persiste indefinidamente)")
    
    with col2:
        st.markdown("### 🔢 Métricas de la red")
        adj_matrix = nx.adjacency_matrix(G).todense()
        degrees = [d for n, d in G.degree()]
        
        st.write("**Grado promedio (〈k〉):**", np.mean(degrees))
        st.write("**Coeficiente de clustering:**", nx.average_clustering(G))
        st.write("**Diámetro:**", nx.diameter(G) if nx.is_connected(G) else "No conectado")
        st.write("**Densidad:**", nx.density(G))
        
        if alpha > 0:
            spectral_radius = max(np.abs(np.linalg.eigvals(adj_matrix)))
            R0_network = beta/alpha * spectral_radius
            st.write(f"**R₀ para redes:** {R0_network:.2f}")
        
        st.write("**Centralidad de intermediación máxima:**", f"{max(nx.betweenness_centrality(G).values()):.3f}")
    
    #Comparación con simulación
    st.markdown("---")
    st.markdown("### 🔄 Comparación Teórica vs Simulación")
    
    sim_final_I = final_counts['I']
    sim_final_S = total_S
    sim_final_R = final_counts['R']
    
    cols = st.columns(3)
    cols[0].metric("Ignorantes (sim)", sim_final_I)
    cols[1].metric("Propagadores (sim)", sim_final_S)
    cols[2].metric("Recuperados (sim)", sim_final_R)
    
    if alpha > 0 and beta > 0:
        theo_final_I = max(0, N_eff - alpha/beta)
        theo_final_S = alpha/beta
        cols[0].metric("Ignorantes (teo)", f"{theo_final_I:.1f}", 
                      delta=round(sim_final_I - theo_final_I, 2))
        cols[1].metric("Propagadores (teo)", f"{theo_final_S:.1f}", 
                      delta=round(sim_final_S - theo_final_S, 2))
    
    #Análisis de sensibilidad
    st.markdown("---")
    st.markdown("### 📉 Análisis de sensibilidad")
    
    #Simular variaciones de beta
    betas = np.linspace(0.1, 1.0, 10)
    final_infected = []
    
    for b in betas:
        temp_G = create_realistic_graph(n_nodes, graph_type)
        for _ in range(steps):
            temp_G = realistic_simulation_step(temp_G, b, alpha)
        counts = sum(1 for n in temp_G.nodes if temp_G.nodes[n]['state'] == 'S')
        final_infected.append(counts)
    
    fig, ax = plt.subplots()
    ax.plot(betas, final_infected, 'r-')
    ax.set_xlabel('Probabilidad de infección (β)')
    ax.set_ylabel('Propagadores finales')
    ax.set_title('Sensibilidad a β')
    st.pyplot(fig)
