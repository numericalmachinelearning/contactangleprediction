# Contact Angle Prediction

The analysis starts with the open-source software blender. The file $\textit{droplet.blend}$ is used to create a 2D animation of a sphere changing shape while dropping into a plane, this represents a raw approximation of a droplet falling on a surface. The animation is composed by 200 frame. This dataset has been used for the test. Using following parameter has been predicted the exact contact angle for each blender image.

$$
\begin{itemize}
 \item \textbf{rx} the semi-major axis
 \item \textbf{ry} the semi-minor axis
 \item \textbf{a} Perpendicular to semi-major axis to the point where the droplet meet the surface.
 \item \textbf{b} Perpendicular to semi-minor axis to the point where the droplet meet the surface. This value has been assumed equal 1.
\end{itemize}
$$

$\frac{a^2}{{r_x}^2}+\frac{b^2}{{r_y}^2} = 1$

```math
SE = \frac{\sigma}{\sqrt{n}}
```
$$
\begin{equation}
 \gamma_{SG} = \gamma_{SL} + \gamma_{LG} cos \theta.
 \label{e0}
\end{equation}
$$

[//]: #![CA](https://user-images.githubusercontent.com/46897230/190351246-d8726b6d-c447-4255-a028-1d42bcb5def7.png)

<img src="https://user-images.githubusercontent.com/46897230/190351246-d8726b6d-c447-4255-a028-1d42bcb5def7.png" width=50% height=50%>
