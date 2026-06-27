# Protocolo de replica: segmentacion de higado y segmentos de Couinaud en MRI

## Contexto

El manuscrito `Contrast-Agnostic Deep Learning for Whole Liver and Couinaud Segment Segmentation in Magnetic Resonance Imaging` no reporta resultados experimentales definitivos. Es una propuesta metodologica: define datos candidatos, preprocesamiento, arquitectura, entrenamiento, evaluacion y cinco experimentos planificados. Por lo tanto, la replica debe tratarse como implementacion y validacion del protocolo, no como reproduccion numerica de tablas ya publicadas.

## Objetivo replicable

Construir una pipeline supervisada 3D para:

1. Segmentacion binaria de higado completo.
2. Segmentacion multic clase de segmentos Couinaud I-VIII.
3. Evaluacion de robustez ante distintos contrastes MRI.
4. Comparacion de variantes con informacion anatomica auxiliar, si existen vasos o landmarks.
5. Calculo de volumetria global y por segmento.

## Datos requeridos

### Imagenes

Volumenes abdominales MRI en NIfTI, DICOM convertido a NIfTI, o formato equivalente soportado por la pipeline actual. Contrastes esperados:

- T1 pre-contraste.
- T1 post-contraste.
- T2.
- Dixon water/fat.
- In-phase y opposed-phase.
- Fase portal venosa, si existe.

### Etiquetas

Minimo:

- Mascara binaria de higado completo.

Ideal:

- Etiqueta multic clase con fondo + Couinaud I-VIII.

Opcional para extensiones:

- Mascaras vasculares.
- Landmarks anatomicos.
- Planos o puntos de referencia hepaticos.
- Metadatos: contraste, vendor, campo magnetico, patologia, institucion, spacing.

## Normalizacion de dataset

Cada caso deberia exponerse a la pipeline con una estructura conceptual similar a:

```text
case_id
image_path
liver_mask_path
couinaud_mask_path
contrast
institution
scanner_vendor
field_strength
pathology_group
spacing
split
```

Si la pipeline del repositorio ya usa un manifiesto CSV/JSON/YAML, conviene extender ese manifiesto con los campos `contrast`, `institution`, `scanner_vendor`, `field_strength` y `pathology_group`. Estos campos son necesarios para reproducir la evaluacion estratificada del paper.

## Preprocesamiento

Pasos alineados con el manuscrito:

1. Convertir todos los volumenes a una orientacion comun.
2. Preservar metadatos fisicos de spacing para calculo volumetrico.
3. Re-muestrear a spacing isotropico o casi isotropico definido por la mediana del dataset.
4. Aplicar clipping robusto por percentiles.
5. Normalizar intensidad por volumen o por crop corporal/hepatico.
6. Aplicar crop abdominal o crop centrado en higado.
7. Opcional: N4 bias field correction, validando que no degrade contraste clinico.
8. Guardar imagenes y mascaras preprocesadas con trazabilidad al caso original.

Para labels, usar interpolacion nearest-neighbor. Para imagenes, usar interpolacion lineal o B-spline segun la convencion de la pipeline.

## Aumentos de datos

Aumentos sugeridos:

- Rotaciones y escalado.
- Deformacion elastica.
- Perturbaciones gamma/intensidad.
- Ruido Gaussiano.
- Blur.
- Bias field simulado.
- Dropout de contraste si hay entradas multicanal.

El objetivo es representar variabilidad MRI plausible, no destruir anatomia.

## Experimento 1: segmentacion de higado completo

### Hipotesis

Un modelo 3D entrenado con datos heterogeneos logra segmentacion robusta de higado completo a traves de multiples contrastes.

### Modelo

Baseline recomendado:

- nnU-Net v2 style 3D.

Baseline alternativo:

- 3D U-Net o residual 3D U-Net si el dataset o GPU son limitados.

### Salida

Mascara binaria: fondo vs higado.

### Evaluacion

Reportar:

- DSC.
- IoU.
- HD95.
- ASSD.
- Volumetric similarity.
- Precision.
- Recall.
- Error absoluto y relativo de volumen.

Estratificar por:

- Contraste.
- Vendor.
- Institucion.
- Campo magnetico.
- Patologia.

## Experimento 2: segmentacion Couinaud I-VIII

### Hipotesis

La segmentacion Couinaud es mas dificil que la segmentacion de higado completo, especialmente para segmentos pequenos o variables como I y IV.

### Estrategias

Comparar:

1. Modelo unificado: una red predice fondo + segmentos I-VIII.
2. Modelo en dos etapas: primera red predice higado, segunda red segmenta Couinaud dentro del higado.

### Salida

Label map multic clase:

```text
0 fondo
1 Couinaud I
2 Couinaud II
3 Couinaud III
4 Couinaud IV
5 Couinaud V
6 Couinaud VI
7 Couinaud VII
8 Couinaud VIII
```

### Evaluacion

Reportar por segmento, no solo promedio:

- DSC.
- IoU.
- HD95.
- ASSD.
- Error relativo de volumen.
- Volumen en ml.

## Experimento 3: generalizacion por contraste

### Disenos

1. Pooled training: entrenar con todos los contrastes disponibles.
2. Leave-one-contrast-out: excluir un contraste durante entrenamiento y probar en ese contraste.
3. Cross-contrast: entrenar en un contraste y probar en otro.

### Comparaciones clave

Medir si el entrenamiento agrupado mejora la robustez o si ciertos contrastes dominan el aprendizaje. La particion debe ser a nivel paciente, no a nivel volumen, para evitar fuga de datos.

## Experimento 4: extensiones anatomicas

Ejecutar solo si los datos lo permiten.

### Variantes

1. Baseline sin informacion auxiliar.
2. Modelo con rama auxiliar de vasos.
3. Modelo con deteccion de landmarks.
4. Modelo con canal de coordenadas normalizadas dentro del higado.

### Criterio de exito

La extension anatomica deberia mejorar bordes intersegmentarios, HD95/ASSD y errores de volumen por segmento, especialmente cerca de planos vasculares.

## Experimento 5: volumetria clinica

### Calculos

Para cada mascara predicha:

- Volumen total de higado.
- Volumen por segmento.
- Porcentaje de cada segmento respecto al volumen hepatico total.
- Si hay plan quirurgico: future liver remnant volume y porcentaje.

### Evaluacion

Reportar:

- Error absoluto de volumen.
- Error relativo de volumen.
- Intraclass correlation coefficient.
- Bland-Altman.
- Acuerdo en decisiones clinicas si existen umbrales definidos.

## Splits recomendados

Prioridad:

1. Train/validation/test por paciente.
2. Test externo si existe otra institucion.
3. Cross-validation de 5 folds si el dataset es pequeno.

Restricciones:

- El mismo paciente no debe aparecer en mas de un split.
- Los contrastes del mismo paciente deben mantenerse en el mismo split.
- Los splits deben conservar diversidad de contraste y patologia.

## Entrenamiento

Configuracion inicial:

- Loss: Dice + cross entropy.
- Optimizer: SGD con Nesterov o AdamW.
- Scheduler: polynomial o cosine.
- Patch-based 3D training.
- Early stopping por DSC de validacion y distancia de superficie.
- Checkpoint elegido sin mirar el test.

Para Couinaud:

- Considerar class weighting o focal loss si segmentos pequenos quedan subrepresentados.
- Considerar boundary-aware loss como experimento secundario si los bordes son inestables.

## Tablas de resultados a generar

### Higado completo

```text
model, test_setting, contrast, DSC, IoU, HD95_mm, ASSD_mm, volume_error_pct, precision, recall
```

### Couinaud

```text
model, test_setting, contrast, segment, volume_ml, DSC, IoU, HD95_mm, ASSD_mm, relative_volume_error_pct
```

### Volumetria

```text
case_id, segment, reference_volume_ml, predicted_volume_ml, absolute_error_ml, relative_error_pct
```

## Mapeo a una pipeline de datos existente

Cuando el repositorio correcto este disponible, revisar estos puntos:

1. Donde define casos/dataset/manifiestos.
2. Como representa imagenes 3D y mascaras.
3. Si ya implementa resampling, clipping, normalization y crop.
4. Si el split es por paciente o por archivo.
5. Si soporta multiples modalidades/canales.
6. Si las metricas actuales incluyen HD95, ASSD y volumetria.
7. Si hay configuraciones por experimento.
8. Si los resultados quedan trazables por contraste y segmento.

## Implementacion minima sugerida

1. Agregar campos de metadatos MRI al manifiesto.
2. Crear un dataset loader que devuelva `image`, `label`, `case_id`, `contrast` y `spacing`.
3. Implementar transformaciones 3D reproducibles.
4. Configurar experimento binario de higado completo.
5. Configurar experimento multic clase Couinaud.
6. Agregar evaluacion estratificada por contraste.
7. Agregar script de volumetria usando spacing real.
8. Exportar CSVs de metricas y predicciones.

## Riesgos principales

- Falta de labels Couinaud completos.
- Ambiguedad inter-observador en planos Couinaud.
- Dataset pequeno o desbalanceado por contraste.
- Fuga de datos si contrastes del mismo paciente se separan entre splits.
- Volumetria incorrecta si se pierden metadatos de spacing.
- Overfitting a un scanner, vendor o protocolo.

## Proxima accion necesaria

El espacio de trabajo actual no contiene el repositorio mencionado. Para adaptar este protocolo a la pipeline real, hay que abrir o montar el repositorio correcto en esta conversacion. Una vez disponible, el siguiente paso es identificar el manifiesto/dataloader existente y proponer cambios concretos de codigo.
