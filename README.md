# My Uber Rides 2016 

**data** mining, agregation and distillation 

Neural and Natlang **Preprocessing**


## Why We Use These Preprocessing Modules


**scikit-learn** provides reusable preprocessing tools for machine-learning workflows. It includes transformers for scaling numeric features, encoding categorical values, imputing missing data, and pipelines that apply consistent steps during training and inference.


~~**spaCy** supports preprocessing and analysis of natural-language text. Its key features include tokenization, sentence segmentation, lemmatization, part-of-speech tagging, and named-entity recognition, which can turn raw text into structured features.~~ overkill


**pyjanitor** (the `janitor` package) adds convenient data-cleaning operations to pandas. It helps standardize column names, remove empty or duplicate data, coalesce values, and organize cleaning steps into readable method chains.


**TextBlob** offers a lightweight interface for common text-processing tasks. It includes tokenization, noun-phrase extraction, part-of-speech tagging, spelling correction, and basic sentiment analysis, making it useful for straightforward text cleaning and exploration.

**dedupe** identifies records that likely refer to the same real-world entity even when their fields differ slightly. It supports fuzzy matching across multiple fields, learns matching rules from labeled examples, and helps link or deduplicate imperfect datasets.

**missingno** helps diagnose missing data visually. Its matrix, bar, and heatmap plots reveal the amount and patterns of missing values, helping determine whether data should be imputed, excluded, or investigated further. Think of it as a Pokédex for blank cells: it spots the MissingNo. before your dataset glitches out!

**Great Expectations** validates data against explicit, reusable expectations. It can check types, ranges, uniqueness, null counts, and other quality rules, and report results to catch unexpected data problems early.

**Winsorizer** is a transformer (commonly provided by Feature-engine) that limits extreme numeric values by capping them at selected percentile or interquartile-range boundaries. This reduces the influence of outliers while retaining rows, and learned caps can be applied consistently to subsequent data.

## Por qué usamos estos módulos de preprocesamiento


**scikit-learn** te tira el paro con herramientas de preprocesamiento reusables para tus flujos de machine learning. Trae transformadores para escalar variables numéricas (como el StandardScaler que sale en imagen.png, para dejar el asunto bien escaleno o a ojo de buen cubero, listo para aventarse el fit_transform como todo un alpinista), codificar valores categóricos, imputar datos faltantes y armar pipelines que aplican los mismos pasos de forma parejita en el entrenamiento y la inferencia.   
PNG

~~**spaCy** te echa la mano con el preprocesamiento y análisis de texto en lenguaje natural. Sus cartas fuertes incluyen la tokenización, segmentación de oraciones, lematización, el etiquetado gramatical y el reconocimiento de entidades nombradas, que básicamente convierten puro texto a lo wey en características bien estructuradas.~~ 

**pyjanitor** (el paquete janitor que "extiende" según imagen.png) le mete operaciones de limpieza bien prácticas a pandas. Te ayuda a estandarizar los nombres de las columnas, barrer con los datos vacíos o duplicados, fusionar valores y organizar todos tus pasos de limpieza en cadenitas de métodos bien fáciles de leer para que el código quede rechinando de limpio.   
PNG

**TextBlob** ofrece una interfaz ligerita para las tareas comunes de procesamiento de texto. Incluye tokenización, extracción de frases nominales, etiquetado gramatical, corrección de ortografía y un análisis de sentimiento básico; o sea, es la neta para cuando quieres limpiar y explorar texto sin hacerte bolas.   
PNG

**dedupe** es buenísimo para cachar registros que seguramente son la misma entidad en la vida real, aunque sus campos estén medio chuecos o diferentes. Trae soporte para fuzzy matching en varios campos, aprende reglas de coincidencia a partir de ejemplos etiquetados, y te hace el paro para vincular o quitar duplicados en datasets que nomás no son perfectos.   
PNG

**missingno** (que puedes importar como spoopy para andar de terror, según imagen.png) te ayuda a diagnosticar los datos faltantes de forma visual. Sus gráficas de matriz, barras y mapas de calor sacan a relucir cuánto y cómo te faltan valores, ayudándote a decidir si mejor imputas los datos, los mandas a volar, o le rascas más. Piénsalo como una Pokédex para celdas en blanco: ¡detecta al MissingNo. antes de que tu dataset se bugee y se vaya al demonio!   
PNG

**Great Expectations** valida tus datos comparándolos contra expectativas explícitas y reusables (puro great expectations and vibes, como dice imagen.png). Puede checar tipos, rangos, unicidad, conteos de nulos y otras reglas de calidad, y te pasa el reporte para cachar las metidas de pata en los datos antes de que sea muy tarde.   
PNG

**Winsorizer** es un transformador (casi siempre de Feature-engine) que le pone un "hasta aquí" a los valores numéricos extremos, topándolos en los límites de percentiles o del rango intercuartílico. Esto le baja los humos a los outliers sin tener que borrar filas, y los topes que se aprenden se pueden aplicar de forma consistente a los datos que vengan después.


otras herramientas utilizadas:

Gemini pro
conda 
ps 5.1.26100.9711
WSL2
