# Contact Angle Prediction

In this repository two machine learning algorithms are used for estimating the contact angle of a droplet on a surface from a set of images. Files are orgnized as follows:<br />
Creation of an image dataset in the file $\textit{dataset.py}$, those images are then blurred and augmente in the file  $\textit{blurred.py}$. The combination of these two form the train dataset for a machine learning algorithm. Examples of $\textbf{Convolutional Neural Network}$ and $\textbf{Random Forest}$ respectively in $\textit{detectCA.py}$ and $\textit{RFca.py}$ files. The blender file $\textit{droplet.blend}$ can be used for the test. The generated images are pre-processed in the file $\textit{crop_lines.py}$. The file $\textit{contact_angles.py}$ contains the parameters for the test analysis used in the machine learning programs.

The analysis starts with the open-source software $\textbf{Blender}$. The file $\textit{droplet.blend}$ is used to create a 2D animation of a sphere changing shape while dropping into a plane, this represents a raw approximation of a droplet falling on a surface. The animation is composed by 200 frame. This dataset has been used for the test. Using following parameter has been predicted the exact contact angle for each blender image.


[//]: #![CA](https://user-images.githubusercontent.com/46897230/190351246-d8726b6d-c447-4255-a028-1d42bcb5def7.png)
<img src="https://user-images.githubusercontent.com/46897230/190351246-d8726b6d-c447-4255-a028-1d42bcb5def7.png" width=50% height=50%>

<d1>
 <dt> $\textbf{rx}$ the semi-major axis
 <dt> $\textbf{ry}$ the semi-minor axis
 <dt> $\textbf{a}$ Perpendicular to semi-major axis to the point where the droplet meet the surface.
 <dt> $\textbf{b}$ Perpendicular to semi-minor axis to the point where the droplet meet the surface. This value has been assumed equal 1.
</dl>

The Ellipse equation can be written as
 
$$
\begin{equation}
 \frac{a^2}{{r_x}^2}+\frac{b^2}{{r_y}^2} = 1
\end{equation}
$$
 
In order to calculate the Contact Angle $\textit{CA}$ we can write

$b = \sqrt{{r_y}^2-{r_y}^2\frac{a^2}{{r_x}^2}}$,  
$\frac{db}{dx} = -\frac{{r_y}^2 a}{{r_x}^2 \sqrt{{r_y}^2-{r_y}^2\frac{a^2}{{r_x}^2}} }$,  
$\frac{db}{dx}(a) = slope$,  
$tg(slope) = \textit{CA}$.

 Clearly $a = r_x \Rightarrow CA = 90^{\circ}$, also when the ellipse (the droplet) is above $r_y/2, \quad CA = 180^{\circ} - CA$.
 
For the training of the two machine learning algorithm in wich each image in the training is associated with the relative contact angle. Starting with the base range of 178 different straight lines representing diffent angles (exluding $0^{\circ}$ and $180^{\circ}$), I created 3 dataset with 20$\times$20 pixels focus area. I applied than the OpenCV Gaussian filter in order to obtain 3 different set of blurred images in which the Contact Angle location is changing in range from 7 to 14 pixels on the horizontal axis. In this way an overall dataset of 1246 images has been obtained. For each image has been applied an averaging filter kernel of three diffrent size. The operation is similar to the one previously seen for the convolution. The filters applied has the shape $3\times3$, $5\times5$, $7\times7$. The larger the size the greater the blur effect (fig.(\ref{gaussian}). The whole new dataset is composed by,

\begin{itemize}
 \item 1246$\times$3 blurred images + 1246 starting images = 4984
\end{itemize}
 
 
```math
SE = \frac{\sigma}{\sqrt{n}}
```
