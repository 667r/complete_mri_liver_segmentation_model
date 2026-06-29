# Protocolo de replica: segmentacion de higado y segmentos de Couinaud en MRI

## Contexto

El manuscrito `Contrast-Agnostic Deep Learning for Whole Liver and Couinaud Segment Segmentation in Magnetic Resonance Imaging` no reporta resultados experimentales definitivos. Es una propuesta metodologica: define datos candidatos, preprocesamiento, arquitectura, entrenamiento, evaluacion y cinco experimentos planificados. Por lo tanto, la replica debe tratarse como implementacion y validacion del protocolo, no como reproduccion numerica de tablas ya publicadas.

Este repositorio contiene la estructura local a utilizar. La cohorte principal esta en `data/Segmentation/Segmentation` como volumenes DICOM, y cada volumen tiene una mascara DICOM de higado completo en su carpeta `masks`. Los contrastes disponibles se definen cruzando `data/SegmentationKey.csv` con `data/SequenceTypes.csv`.

El subconjunto `data/8 segmentos` no debe tratarse como cobertura completa del dataset: contiene mascaras NIfTI de segmentos Couinaud solo para algunos pacientes y solo para algunos contrastes.

## Objetivo replicable

Construir una pipeline supervisada 3D para:

1. Segmentacion binaria de higado completo.
2. Evaluacion de robustez ante los contrastes MRI realmente presentes en el dataset local.
3. Preparacion opcional de un dataset multic clase de segmentos Couinaud I-VIII con el subconjunto NIfTI disponible.
4. Calculo de volumetria global y, solo cuando exista mascara Couinaud, volumetria por segmento.

## Datos requeridos

### Imagenes

Volumenes abdominales MRI en DICOM bajo:

```text
data/Segmentation/Segmentation/{paciente}/{serie}/images/*.dicom
```

Para entrenamiento con nnU-Net, estos DICOM deben convertirse a NIfTI en la estructura `imagesTr` con la convencion `{case_id}_0000.nii.gz`.

### Contrastes actuales

La referencia canonica de contraste es:

1. `data/SegmentationKey.csv`: mapea `DLDS, Series` a una etiqueta corta.
2. `data/SequenceTypes.csv`: mapea esa etiqueta corta al nombre del contraste.

Contrastes presentes con imagen DICOM y mascara DICOM de higado completo:

```text
Label,Series Description,volumenes
B,Axial Precontrast Fat Suppressed T1w (dynpre),85
C,Mid Arterial T1w,3
E,Axial Late Dynamic T1w,2
G,Axial In Phase (t1nfs),82
H,Axial Opposed Phase (opposed),54
K,Portal Venous T1w (dynportal),83
O,Early Arterial T1w,1
```

En total: 310 volumenes DICOM de 95 pacientes, todos con carpeta `images` y `masks`.

### Etiquetas

Minimo:

- Mascara binaria de higado completo en DICOM:

```text
data/Segmentation/Segmentation/{paciente}/{serie}/masks/*.dicom
```

Subconjunto disponible:

- Mascaras NIfTI de segmentos Couinaud I-VIII en `data/8 segmentos`.
- Cobertura parcial: 197 mascaras NIfTI, 73 pacientes.
- Contrastes con mascara Couinaud parcial:

```text
Label,Series Description,mascaras Couinaud
B,Axial Precontrast Fat Suppressed T1w (dynpre),56
C,Mid Arterial T1w,3
E,Axial Late Dynamic T1w,2
G,Axial In Phase (t1nfs),50
H,Axial Opposed Phase (opposed),26
K,Portal Venous T1w (dynportal),59
O,Early Arterial T1w,1
```

Opcional para extensiones si aparece en datos externos:

- Mascaras vasculares.
- Landmarks anatomicos.
- Planos o puntos de referencia hepaticos.
- Metadatos: contraste, vendor, campo magnetico, patologia, institucion, spacing.

## Normalizacion de dataset

Cada caso deberia exponerse a la pipeline con una estructura conceptual similar a:

```text
case_id
patient_id
series_id
image_dicom_dir
liver_mask_dicom_dir
couinaud_mask_path
contrast_label
contrast_name
spacing
split
```

Para nnU-Net v2, cada volumen debe quedar como un caso independiente de un solo canal MRI:

```text
nnUNet_raw/Dataset001_Liver/
dataset.json
imagesTr/{case_id}_0000.nii.gz
labelsTr/{case_id}.nii.gz
```

Los campos `patient_id`, `series_id`, `contrast_label` y `contrast_name` deben conservarse en un manifiesto externo para analisis estratificado. El mismo paciente no debe dividirse entre train, validacion y test aunque tenga varios contrastes.

Para labels, usar interpolacion nearest-neighbor. Para imagenes, usar interpolacion lineal o B-spline segun la convencion de la pipeline.

## Aumentos de datos

Aumentos sugeridos:

- Rotaciones 3D leves.
- Escalado isotropico leve.

Aplicar la misma transformacion espacial a imagen y label. Para labels usar interpolacion nearest-neighbor; para imagenes usar interpolacion lineal. No usar aumentos de intensidad como parte del protocolo minimo salvo que se definan como experimento adicional, porque el objetivo actual es aislar la variabilidad geometrica propuesta.

## Experimento 1: segmentacion de higado completo

### Hipotesis

Un modelo 3D entrenado con los volumenes DICOM locales logra segmentacion robusta de higado completo a traves de los contrastes B, C, E, G, H, K y O.

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
- Precision.
- Recall.
- Error absoluto y relativo de volumen.

Estratificar por:

- Contraste.
- Paciente.
- Campo magnetico solo si se extrae de metadatos DICOM y queda disponible en el manifiesto.

## Experimento 2: segmentacion Couinaud I-VIII

### Hipotesis

La segmentacion Couinaud es mas dificil que la segmentacion de higado completo, especialmente para segmentos pequenos o variables como I y IV. En este repositorio, este experimento debe tratarse como secundario porque las mascaras Couinaud existen solo para un subconjunto de pacientes y contrastes.

### Estrategias

Estrategia local recomendada:

1. Preparar un Dataset002_LiverSegments solo con pares imagen DICOM + mascara Couinaud NIfTI existentes.
2. Mantener split por paciente, no por volumen.
3. Evaluar por contraste, pero reportar claramente la cobertura parcial.
4. Dejar el modelo en dos etapas como extension futura si la mascara de higado completo se usa para recortar o restringir la prediccion Couinaud.

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
- Error relativo de volumen.
- Volumen en ml.

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
2. Cross-validation de 5 folds con splits manuales por paciente si se usa nnU-Net.
3. Test externo solo si aparece otra institucion o cohorte independiente.

Restricciones:

- El mismo paciente no debe aparecer en mas de un split.
- Los contrastes del mismo paciente deben mantenerse en el mismo split.
- Los splits deben conservar diversidad de contraste y patologia.
- Para nnU-Net, todos los casos etiquetados de entrenamiento/validacion quedan en `imagesTr` y `labelsTr`; la separacion validacion se controla con splits manuales o con la cross-validation interna.
- Si se define test holdout, guardar imagenes en `imagesTs` y conservar labels en un directorio/manifiesto separado para evaluacion propia, porque nnU-Net no usa `labelsTs` durante inferencia.

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

En este repositorio, revisar estos puntos antes de entrenar:

1. Que la conversion DICOM a NIfTI preserve spacing, origen y orientacion.
2. Que cada label tenga la misma geometria que su imagen.
3. Que `dataset.json` use `channel_names: {"0": "MRI"}` y labels consecutivos.
4. Que el manifiesto de casos conserve `patient_id`, `series_id`, `contrast_label` y `contrast_name`.
5. Que los splits sean por paciente.
6. Que las metricas exportadas incluyan contraste y, en Couinaud, segmento.
7. Que HD95, ASSD y volumetria se calculen con spacing real.
8. Que las mascaras Couinaud NIfTI se usen solo cuando exista correspondencia con la serie DICOM.

## Implementacion minima sugerida

1. Crear referencia cruzada desde `SegmentationKey.csv` y `SequenceTypes.csv`.
2. Generar dataset nnU-Net de higado completo desde DICOM hacia `imagesTr` y `labelsTr`.
3. Generar splits por paciente y un manifiesto trazable.
4. Implementar aumentacion 3D reproducible con rotacion y escalado.
5. Configurar experimento binario de higado completo.
6. Preparar Dataset002 Couinaud solo con las mascaras NIfTI disponibles.
7. Agregar evaluacion estratificada por contraste.
8. Agregar script de volumetria usando spacing real.
9. Exportar CSVs de metricas y predicciones.

## Riesgos principales

- Falta de labels Couinaud completos.
- Ambiguedad inter-observador en planos Couinaud.
- Desbalance fuerte por contraste: C, E y O tienen muy pocos casos.
- Fuga de datos si contrastes del mismo paciente se separan entre splits.
- Volumetria incorrecta si se pierden metadatos de spacing.
- Overfitting a un scanner, vendor o protocolo.
- Posible desalineacion geometrica al combinar imagenes DICOM con mascaras Couinaud NIfTI.
- Confundir la cobertura DICOM completa de higado con la cobertura parcial de Couinaud.

## Proxima accion necesaria

Acciones de preparacion, sin entrenar:

1. Ejecutar la referencia cruzada para auditar contrastes y cobertura.
2. Revisar el manifiesto generado antes de convertir DICOM a NIfTI.
3. Ejecutar la preparacion nnU-Net solo cuando se confirme el directorio de salida.
4. Ejecutar la aumentacion solo sobre un dataset nnU-Net ya validado.
5. Antes de entrenar, correr `nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity`.

## Scripts agregados

Referencia cruzada de contrastes:

```bash
python src/cross_reference_contrasts.py --output-dir data/reports
```

Preparacion nnU-Net para higado completo, sin escribir archivos:

```bash
python src/prepare_nnunet_dataset.py --task whole-liver --output-root nnUNet_raw --dry-run
```

Preparacion nnU-Net para el subconjunto Couinaud, sin escribir archivos:

```bash
python src/prepare_nnunet_dataset.py --task couinaud --output-root nnUNet_raw --dry-run
```

Aumentacion geometrica sobre un dataset nnU-Net ya preparado:

```bash
python src/augment_nnunet_dataset.py --dataset-dir nnUNet_raw/Dataset001_Liver --output-dir nnUNet_raw/Dataset101_LiverAug --dry-run
```

Exportacion de metricas nnU-Net al formato esperado del protocolo:

```bash
python src/export_nnunet_protocol_results.py --input models --output-file data/reports/nnunet_protocol_results.csv
```
