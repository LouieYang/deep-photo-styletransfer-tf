# deep-photo-styletransfer-tf

This is a pure Tensorflow implementation of [Deep Photo Styletransfer](https://arxiv.org/abs/1703.07511)

This implementation support [L-BFGS-B](https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface) (which is what the original authors used) and [Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) in case the ScipyOptimizerInterface incompatible when Tensorflow upgrade to higher version.

This implementation may seems a lot simpler thanks to Tensorflow's [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

Additionally, there is no dependency on MATLAB thanks to another [repository](https://github.com/martinbenson/deep-photo-styletransfer/blob/master/deep_photo.py) compute Matting Laplacian Sparse Matrix. Below is example of transferring the photo style to another photograph.

<p align="center">
    <img src="./some_results/best5.png" width="512"/>
    <img src="./examples/readme_examples/intar5.png" width="290"/>
</p>

### Examples
Here are more result from tensorflow algorithm (from left to right are input, style, torch results and tensorflow results)

<p align="center">
    <img src='examples/input/in6.png' height='140' width='210'/>
    <img src='examples/style/tar6.png' height='140' width='210'/>
    <img src='examples/final_results/best6_t_1000.png' height='140' width='210'/>
    <img src='some_results/best6.png' height='140' width='210'/>
</p>

<p align="center">
    <img src='examples/input/in7.png' height='140' width='210'/>
    <img src='examples/style/tar7.png' height='140' width='210'/>
    <img src='examples/final_results/best7_t_1000.png' height='140' width='210'/>
    <img src='some_results/best7.png' height='140' width='210'/>
</p>

<p align="center">
    <img src='examples/input/in8.png' height='140' width='210'/>
    <img src='examples/style/tar8.png' height='140' width='210'/>
    <img src='examples/final_results/best8_t_1000.png' height='140' width='210'/>
    <img src='some_results/best8.png' height='140' width='210'/>
</p>

<p align="center">
    <img src='examples/input/in9.png' height='140' width='210'/>
    <img src='examples/style/tar9.png' height='140' width='210'/>
    <img src='examples/final_results/best9_t_1000.png' height='140' width='210'/>
    <img src='some_results/best9.png' height='140' width='210'/>
</p>

<p align="center">
    <img src='examples/input/in10.png' height='140' width='210'/>
    <img src='examples/style/tar10.png' height='140' width='210'/>
    <img src='examples/final_results/best10_t_1000.png' height='140' width='210'/>
    <img src='some_results/best10.png' height='140' width='210'/>
</p>

<p align="center">
    <img src='examples/input/in11.png' width='210'/>
    <img src='examples/style/tar11.png' width='210'/>
    <img src='examples/final_results/best11_t_1000.png' width='210'/>
    <img src='some_results/best11.png' width='210'/>
</p>
