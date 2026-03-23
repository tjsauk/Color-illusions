# Color Illusions Toolkit

Exploratory tools for investigating various aspects of color perception and visual illusion phenomena.

The tools in this repository were developed to enable interactive experimentation with perceptual categorization, chromatic ambiguity, illumination effects, and contextual color consistency. The results obtained using these tools are based on qualitative perceptual observations and may vary across observers and viewing conditions.

---

## Basecolors.py

This tool provides a framework for experimenting with definitions of base colors within the RGB color space. Colors are represented as vectors originating from RGB black and parameterized by:

- vector length (color richness / chromatic strength)
- tilt (measured from grey)
- azimuth (angle around grey)

Users can adjust these parameters to generate color mixtures and mappings.

In the conducted exploratory experiments, base colors were determined using the following criterion:

- Base colors must be clearly distinguishable from each other.
- Variations in vector length or tilt should not alter the perceived categorical identity of a base color until the color becomes achromatic (i.e., approaches the grey axis between black and white).
- Neighboring base colors must remain perceptually distinct.
- The number of base colors was increased incrementally until mid-way chromatic mixtures could be consistently categorized as belonging to one of the defined base colors.


---

## Ambiguous color border regions.py

This tool allows exploration of perceptual ambiguity between neighboring base colors.

The lower visualization window displays an unfolded cone segment of the RGB color space defined between selected base colors. Users can define a boundary angle measured from the base color with greater vector length (under equal luminance conditions).

In exploratory observations, setting this boundary angle to approximately 12 degrees enabled users to experience perceptual ambiguity near the boundary region. Colors close to this boundary may appear to shift categorization between adjacent base colors.

Masking one side of the boundary removes the ambiguity, causing all observed shades to appear as variations of the remaining base color.

Perception of this ambiguity might depend on the user’s internal definition of base color and the settings used.

---

## Ambiguous color to grey border regions.py

This tool enables similar exploration of ambiguity between chromatic base colors and achromatic (grey) perception.

In this case, the boundary is defined by radial distance from the grey axis in RGB space. Users may need to rotate the visualization to observe the relevant triangular segment of color space.

The experimental procedure mirrors that used for chromatic boundary exploration.

---

## Lighting illusion.py

This tool allows exploration of perceived color changes resulting from variations in luminance and spatial configuration.

Example parameters that produce observable effects:

- Large patch strength: 0.6
- Small patch strength: 0.4
- Small patch size: 0.35

Under these conditions, the smaller patch may appear darker relative to a reference region despite minor physical differences.

Increasing the size of the smaller patch to approximately match the visible area of the larger patch reduces the effect. Similarly, increasing the strength of the smaller patch relative to the larger one alters the perceptual outcome.

This illusion is generally more difficult to produce reliably compared to other explored phenomena.

---

## Color consistency playground.py

This tool supports exploration of scene-level chromatic consistency illusions.

Suggested usage:

1. Load an image and inspect the 3D distribution of sampled pixel colors (by keeping only the Original selected int the Show point sets).
2. If pixel clusters lie near the grey axis (approximately lie with in a radius halfway to the basecolors and grey when viewed from the top), choose a tint direction opposite to the target perceived color shift.
3. Example: Red (azimuth ~30°) shifted toward cyan (azimuth ~210°) produces a strong observable effect.
4. Adjust tint vector length until all target-colored pixels cross the grey axis (by choosing in Show point sets only the Tinted).
5. Apply compensation using the reverse azimuth to approximate perceived reconstruction.

The tool displays:

- the tinted image (stimulus)
- an approximation of perceived color reconstruction (not a physical reversal)

Additional experimental modes:

- projection of colors onto the tint–compensation plane. Removes the illusion colors that do not exist in the plane.
- mirroring across the same plane to explore perceptual shifts, changes the illusions to happen with clearly wrong colors.

Illusion observability depends strongly on the distribution of image colors relative to the grey axis.

Images similar in chromatic structure to those used in canonical color reconstruction illusions tend to produce clearer effects.
