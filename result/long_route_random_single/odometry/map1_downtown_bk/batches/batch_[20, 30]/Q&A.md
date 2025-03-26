
used frames [20, 30]

The images I uploaded contains 10 frames of images from a vehicle dash cam video. 
Determine the vehicle's movement direction (delta heading) and displacement (delta distance) between each frame. That is, for the 10 I uploaded, you should be able to generate 9 pairs of direction and displacement.
For delta heading, assume the vehicle's starting position in the first frame is heading 0 degree.You should then represent all later delta headings in degree values ranging between -180 and 180 degrees(clockwise) relative to the starting position at 0 degree.
Here, -180 and +180 degree would mean turning around and go in the complete opposite way compared to the previous frame.
For displacement, give your answer values in meter unit.
Finally, present your response in json format:
```json {"delta_heading": [degree1, degree2, ... degree9],
"displacement": [displacement1, displacement2, ... displacement9]
}```
Make sure you have 9 int or float elements in both the direction list and displacement list.
The delta headings and displacements can't be all the same

```json
{
  "delta_heading": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0
  ],
  "displacement": [
    10.0,
    12.0,
    14.0,
    11.0,
    12.0,
    13.0,
    11.0,
    12.0,
    14.0
  ]
}
```