---
sidebar_position: 2
---

# Parameters
## UAV-Orthomosaik
"Choose an UAV-Orthomosaic of a windthrow area which should be analyzed. The provided models work on an internal pixel size of 3 cm. If the pixel size of the analyzed orthomosaic which should be analyzed is significantly greater, the detection rate will decrease dramatically"
ggf. Abfrage der Pixelgröße und Warnung bei einer px> 5cm, dass es sich nicht um ein typisches UAV-Orthomosaik handelt und aufgrund der Pixelgröße die Erkennungsraten deutlich nach unten gehen und ermittelten Voluminas nicht verlässlich sind.

## Model:
"Choose a model for the semantic segmentation of the UAV-orthomosaics. Tree species specific models are provided for beech and spruce and a general model for mixed stands or other tree species."
dropdown mit 3 trainierte Modell zur Auswahl. Mit Option ein selbsttrainiertes Model zu verwenden (File Dialoge, Abfrage Datentyp hd5)

## Optionen Stem Detection:
 * `min length = 2 (float)`: "Lower threshold for the length of a stem to be recognized"
 * `max distance =8 m [<max tree height]  (float)`: "Maximum distance between two stem segments to be connected in the stem reconstruction. Set the threshold according to the maximum tree crown diameter of the analyzed stand.
 * `tolerance angle = 7 ° [<30°] (float)`: "Maximum angle between two stem segments to be connected in the stem reconstruction"
 * `max tree height = 32 m [>min length] (float)`: "Upper threshold for the maximum stem length in the reconstruction process. Higher threshold will slow down the stem reconstruction process. Set it close to the maximum tree height to get ideal results."

## Optionen Semantic Segmentation:
Maybe deactivated when selecting a provided model
"Only change accordingly when applying a custom model" or maybe simply deactivated when selecting a provided model
tile size = 15 m "Corresponding side length of the training tiles the model was trained with"
image width = 512 (int) "Size of the training tiles in pixels"
