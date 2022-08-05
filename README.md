# kProp
$kProp$ is a neural network verifier for ReLU neural networks with adversarial distortions in general norms. It employs the multi-neuron relaxation to capture the relations of neurons in the same layer, and region clipping to filter out the infeasible inputs.

## User Manmul
### Installation
First clone this repository via git as follows:
```bash
git clone https://github.com/xue-xy/kProp.git
cd kProp
```
Then install the python dependencies:
```bash
pip install -r requirements.txt
```
### Usage
```bash
python run.py --model <model name> -n <norm> -r <radius> -k <group numeber>
```
+ `<model>`: the model you want to check.
+ `<n>`: norm, string. Now only 1, 2, and $\infty$ support region clipping.
+ `<r>`: radius, float between 0 and 1.
+ `<k>`: number of neurons in a neuron group. Joint bound are computed for each group.