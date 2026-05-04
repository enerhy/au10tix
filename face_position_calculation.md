# Face Position Calculation

This document explains how the face geometry attributes are calculated from the facial landmarks.

The calculations use five facial landmarks:

- left eye: `landmarks[0]`
- right eye: `landmarks[1]`
- nose: `landmarks[2]`
- left mouth corner: `landmarks[3]`
- right mouth corner: `landmarks[4]`

## Interocular Distance

The **interocular distance** measures the distance between the left and right eye landmarks.

It is calculated using the Euclidean distance between the two eye points:

```python
interocular_distance = distance(left_eye, right_eye)
```

This value represents the face size in pixels. Larger values usually indicate that the face is closer to the camera, while smaller values indicate a smaller or more distant face.

## Yaw Estimate

The **yaw estimate** approximates how much the face is turned left or right.

It compares the horizontal distance from the nose to each eye:

```python
left_dist = abs(nose_x - left_eye_x)
right_dist = abs(right_eye_x - nose_x)

yaw = (left_dist - right_dist) / interocular_distance
```

If the nose is centered between the eyes, the yaw value is close to `0`.

If the value becomes larger in magnitude, the face is likely turned more to one side.

## Roll Angle

The **roll angle** measures how much the face is tilted clockwise or counterclockwise.

It is calculated from the slope of the line between the two eyes:

```python
dx = right_eye_x - left_eye_x
dy = right_eye_y - left_eye_y

roll_angle = arctan2(dy, dx)
```

The result is converted from radians to degrees.

A roll angle close to `0` means the eyes are nearly horizontal.

A larger positive or negative value means the face is tilted.

## Mouth-to-Nose Vertical Ratio

The **mouth-to-nose vertical ratio** measures the vertical distance between the nose and the center of the mouth, normalized by the interocular distance.

First, the mouth center is estimated from the two mouth corner landmarks:

```python
mouth_center_y = (left_mouth_y + right_mouth_y) / 2
```

Then the vertical distance from the nose to the mouth center is divided by the interocular distance:

```python
ratio = abs(mouth_center_y - nose_y) / interocular_distance
```

Normalizing by interocular distance makes the value less dependent on the absolute size of the face in the image.

