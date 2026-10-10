# Initialisierungsfunktionen für neuronale Netze

## Konstant
- Zeros
- Ones
- Constant
- Identity
- Eye
- Dirac
- Delta-Orthogonal
- Fill

## Zufällig, Basisverteilungen
- Uniform
- Normal (Gaussian)
- TruncatedNormal
- LogNormal
- Laplace
- Cauchy
- Bernoulli
- Sparse
- RandomUniform
- RandomNormal

## Fan-basiert (Varianzskalierung)
- VarianceScaling
- Xavier Uniform (Glorot Uniform)
- Xavier Normal (Glorot Normal)
- He Uniform (Kaiming Uniform)
- He Normal (Kaiming Normal)
- LeCun Uniform
- LeCun Normal
- Fan-In
- Fan-Out
- Fan-Avg
- Kaiming Uniform (fan_in)
- Kaiming Uniform (fan_out)
- Kaiming Normal (fan_in)
- Kaiming Normal (fan_out)
- Calculate Gain
- Scaled Residual Init (GPT-2)
- Fixup
- ReZero
- LayerScale Init
- SkipInit
- T-Fixup
- DeepNet Init (DeepNorm)
- Mitchell Init
- Megatron Init
- Small Init
- Wang Init
- muP (Maximal Update Parametrization)

## Orthogonal und strukturiert
- Orthogonal
- Semi-Orthogonal
- Delta-Orthogonal
- Unitary
- Haar-Orthogonal
- Householder
- Givens
- Hadamard
- Fourier (DFT)
- Circulant
- Toeplitz
- Block-Diagonal
- Low-Rank
- Butterfly
- Kronecker
- Dirac

## Aktivierungsspezifisch
- SIREN Init
- SELU Init (LeCun Normal)
- Swish Init
- GELU Init
- Snake Init
- Fourier Features Init
- Random Fourier Features
- Gabor Init
- Wavelet Init
- Positional Encoding Init (Sinus/Cosinus)
- RBF Center Init
- K-Means Init
- Periodic Init
- Sparse Init (Martens)
- Echo State Init
- Spectral Radius Init

## Datenabhängig
- LSUV (Layer-Sequential Unit-Variance)
- Data-Dependent Init
- Data-Dependent ActNorm Init
- Weight-Norm Data Init
- PCA Init
- SVD Init
- ICA Init
- Whitening Init
- K-Means Init
- Gradient-Norm Init (GradInit)
- ZerO Init
- Mimetic Init
- SAL (Signal Propagation) Init
- Dynamical Isometry Init
- Edge-of-Chaos Init
- NTK Init
- Hessian-Aware Init
- Meta-Learned Init (MAML)
- Reptile Init
- Pretrained Init
- Transfer Init
- Net2Net Init
- Network Morphism Init
- Weight Selection Init
- Weight Inheritance Init
- Lottery-Ticket Init
- Rewinding Init

## Bias
- Zero Bias
- Constant Bias
- Prior Bias (Focal Loss)
- Forget-Gate Bias (LSTM)
- Output Bias aus Klassenverteilung
- Log-Prior Bias
- Mean-Target Bias
- Uniform Bias
- Normal Bias

## Rekurrente Netze
- Orthogonal Recurrent
- Identity Recurrent (IRNN)
- Chrono Init
- Unitary RNN Init
- Echo State Init
- Spectral Radius Scaling
- Forget-Gate Bias
- Glorot Input / Orthogonal Recurrent
- HiPPO Init
- S4 Init (DPLR)
- S4D Init
- S5 Init
- Mamba Init (A-Log, Delta)
- LRU Init
- Legendre Memory Unit Init

## Faltung
- Delta-Orthogonal
- Dirac
- Orthogonal Conv
- Fan-Out Kaiming Conv
- Bilinear Upsampling Init
- Nearest-Neighbor Upsampling Init
- Gabor Filter Init
- Sobel Init
- Gaussian Filter Init
- Haar Wavelet Init

## Embedding und Aufmerksamkeit
- Normal (std=0.02)
- Xavier Uniform Embedding
- Sinusoidal Positional Init
- Learned Positional Init
- RoPE Init
- ALiBi Init
- Relative Position Bias Init
- Zero Init (Output Projection)
- Scaled Init (1/sqrt(2·L))
- QK-Norm Init
- Attention Temperature Init
- Pretrained Embedding Init (word2vec, GloVe, fastText)
- Tied Embedding Init

## Normierung
- Gamma Ones
- Beta Zeros
- Gamma Zero (Zero-Gamma für Residual)
- BatchNorm Running Mean Zeros
- BatchNorm Running Var Ones
- LayerNorm Weight Ones
- RMSNorm Weight Ones
- GroupNorm Weight Ones
- Weight-Norm g Init
- Spectral-Norm u Init

## Quantisierungsbewusst
- Scale Init (MinMax)
- Scale Init (Percentile)
- Scale Init (MSE)
- Scale Init (Entropy)
- LSQ Step Size Init
- LSQ+ Offset Init
- PACT Alpha Init
- Zero-Point Init
- Codebook Init (K-Means)
- Binary Weight Init
- Ternary Weight Init

## Adapter und Fine-Tuning
- LoRA A (Kaiming Uniform)
- LoRA B (Zeros)
- DoRA Magnitude Init
- PiSSA Init
- MiLoRA Init
- LoRA-GA Init
- OLoRA Init
- Adapter Near-Identity Init
- Prefix Init
- Prompt Embedding Init
- IA3 Init (Ones)
- BitFit Init
- Zero-Init Gate (Flamingo, LLaMA-Adapter)

## Spezielle Architekturen
- GAN Init (Normal 0, 0.02)
- DCGAN Init
- StyleGAN Init
- Diffusion Zero-Init Output
- UNet Zero-Init Output
- ResNet Zero-Init-Residual
- Transformer Init (Vaswani)
- BERT Init (std 0.02)
- GPT-2 Init
- GPT-NeoX Init
- T5 Init
- LLaMA Init
- ViT Init (Truncated Normal)
- Swin Init
- Highway Init (Transform-Gate Bias negativ)
- DenseNet Init
- MobileNet Init
- EfficientNet Init (Fan-Out Uniform)
- Capsule Init
- Hypernetwork Init
- Neural ODE Init
- Spiking Network Init
- Hopfield Init
- Boltzmann Machine Init
- Autoencoder Tied Init
- RBM Pretraining Init
- Greedy Layer-wise Pretraining
- Denoising Autoencoder Pretraining

## Klassische Verfahren
- Small Random (±0.01)
- Uniform (−1/√n, 1/√n)
- Uniform (−0.05, 0.05)
- Nguyen-Widrow
- Sparse Initialization
- Marginal Init
- Mean-Field Init
- Gaussian Prior Init
- Bayesian Prior Init
- Variational Posterior Init
- Weight Perturbation Init
- Noisy Init
- Symmetry-Breaking Init
