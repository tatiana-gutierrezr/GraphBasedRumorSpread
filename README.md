# 🌍 Simulación de Propagación de Rumores en Redes Sociales

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![NetworkX](https://img.shields.io/badge/NetworkX-%23007ACC.svg?style=for-the-badge&logo=networkx&logoColor=white)](https://networkx.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

Una simulación interactiva que modela cómo se propagan los rumores en diferentes tipos de redes sociales, incorporando comportamientos humanos realistas.

## 🚀 Características principales

- **Modelado de redes complejas**: Simula 4 tipos de redes sociales diferentes:
  - Redes comunitarias (con parámetros configurables)
  - Modelo Watts-Strogatz (redes de mundo pequeño)
  - Modelo Barabasi-Albert (redes libres de escala)
  - Modelo Erdős-Rényi (redes aleatorias)

- **Comportamientos realistas**:
  - Usuarios inmunes (nunca creen el rumor)
  - Propagadores permanentes (nunca dejan de difundir)
  - Usuarios escépticos (difíciles de convencer)
  - Usuarios olvidadizos (se recuperan rápidamente)

- **Análisis matemático avanzado**:
  - Modelo SIR mejorado con ecuaciones diferenciales
  - Retratos de fase y análisis de estabilidad
  - Comparación teoría vs simulación
  - Análisis de sensibilidad a parámetros

- **Visualización interactiva**:
  - Animación en tiempo real de la propagación
  - Evolución temporal de los estados
  - Código de colores para diferentes tipos de usuarios

## 📦 Requisitos e instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tatiana-gutierrezr/GraphBasedRumorSpread.git
```

2. Crea y activa un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta la aplicación:
```bash
streamlit run rumorspreading.py
```

## 🛠️ Dependencias principales

- Python 3.8+
- Streamlit (interfaz web)
- NetworkX (modelado de redes)
- Matplotlib (visualización)
- NumPy (cálculos científicos)
- SciPy (ecuaciones diferenciales)
- Pandas (análisis de datos)

## 🎨 Cómo usar la aplicación

1. **Configura la red social**:
   - Selecciona el tipo de red (comunitaria, Watts-Strogatz, etc.)
   - Ajusta parámetros específicos de cada red
   - Define el número de usuarios (nodos)

2. **Configura los comportamientos**:
   - Probabilidad de infección (β) y recuperación (α)
   - Porcentajes de usuarios especiales (inmunes, escépticos, etc.)

3. **Ejecuta la simulación**:
   - Observa la propagación en tiempo real
   - Analiza los resultados finales
   - Explora el análisis matemático en la pestaña correspondiente

## 📊 Resultados y análisis

La aplicación proporciona:
- Gráficos de evolución temporal
- Estadísticas detalladas de cada tipo de usuario
- Comparación con modelos teóricos
- Análisis de estabilidad del sistema
- Retratos de fase interactivos

## 📚 Marco teórico

El proyecto implementa un modelo SIR (Susceptible-Infectado-Recuperado) extendido con:

- **Ecuaciones diferenciales**:
  ```
  dS/dt = -β(SI/N)
  dI/dt = β(SI/N) - αI
  dR/dt = αI
  ```

- **Extensiones para comportamientos realistas**:
  - Umbrales individuales de infección
  - Tasas de recuperación variables
  - Estados especiales (inmunes, propagadores permanentes)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor abre un issue o envía un pull request con tus sugerencias.

---

Desarrollado por [Tatiana Gutierrez R] | [2025]
```
