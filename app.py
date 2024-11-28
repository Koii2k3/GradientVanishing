import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go


# Define a simple deep neural network
class SimpleNN(nn.Module):
    def __init__(self, num_layers=10, activation="sigmoid", init_weights="normal", norm_type=None, use_skip=False):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.norm_type = norm_type
        self.use_skip = use_skip
        self.norm_layers = nn.ModuleList()

        # Add layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(10, 10))
            if norm_type == "BatchNorm":
                self.norm_layers.append(nn.BatchNorm1d(10))
            elif norm_type == "LayerNorm":
                self.norm_layers.append(nn.LayerNorm(10))

        self.init_weights(init_weights)

    def init_weights(self, method):
        for layer in self.layers:
            if method == "normal":
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        skip_connection = x
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply normalization
            if self.norm_type is not None:
                x = self.norm_layers[i](x)

            # Apply activation function
            if self.activation == "sigmoid":
                x = torch.sigmoid(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(x, negative_slope=0.01)
            elif self.activation == "swish":
                x = x * torch.sigmoid(x)

            # Add skip connection
            if self.use_skip and i % 2 == 1:
                x = x + skip_connection
                skip_connection = x

        return x


# Streamlit app
st.title("Advanced Gradient Vanishing Demo")

st.sidebar.header("Settings")

# Number of layers
num_layers = st.sidebar.slider(
    "Number of Layers", min_value=3, max_value=30, value=10, step=1)

# Activation function
activation = st.sidebar.selectbox(
    "Activation Function", ["sigmoid", "tanh", "relu", "leaky_relu", "swish"]
)

# Weight initialization
init_weights = st.sidebar.selectbox(
    "Weight Initialization", ["normal", "xavier", "he"]
)

# Normalization
norm_type = st.sidebar.selectbox(
    "Normalization Type", ["None", "BatchNorm", "LayerNorm"]
)

# Skip connection
use_skip = st.sidebar.checkbox("Use Skip Connection")

# Optimizer
optimizer_type = st.sidebar.selectbox(
    "Optimizer", ["SGD", "Adam"]
)

# Learning rate
learning_rate = st.sidebar.slider(
    "Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001)

# Input data distribution
input_type = st.sidebar.selectbox(
    "Input Data Distribution", ["Gaussian (Normal)", "Uniform"]
)

# Batch size
batch_size = st.sidebar.slider(
    "Batch Size", min_value=1, max_value=100, value=32)

# Generate input data
if input_type == "Gaussian (Normal)":
    input_data = torch.randn(batch_size, 10)
else:  # Uniform
    input_data = torch.rand(batch_size, 10)

# Create model
model = SimpleNN(num_layers=num_layers, activation=activation, init_weights=init_weights,
                 norm_type=norm_type if norm_type != "None" else None, use_skip=use_skip)

# Select optimizer
if optimizer_type == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_type == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Forward pass to compute activations
output = model(input_data)

# Compute gradients
loss = torch.sum(output)  # Dummy loss
optimizer.zero_grad()
loss.backward()

# Extract gradients
gradients = []
for i, layer in enumerate(model.layers):
    if layer.weight.grad is not None:
        gradients.append(layer.weight.grad.norm().item())
    else:
        gradients.append(0)

# Interactive plot using Plotly
fig = go.Figure()

# Gradient plot with annotations
fig.add_trace(
    go.Scatter(
        x=list(range(1, len(gradients) + 1)),
        y=gradients,
        mode="lines+markers+text",
        text=[f"{g:.4f}" for g in gradients],
        textposition="top center",
        name="Gradient Norm",
        line=dict(color="blue"),
    )
)

fig.update_layout(
    title="Gradient Norm Across Layers",
    xaxis_title="Layer",
    yaxis_title="Gradient Norm",
    legend=dict(yanchor="top", y=1, xanchor="right", x=1),
    hovermode="x unified",
)

st.plotly_chart(fig)

# Observations
st.markdown("### Observations:")
st.markdown(
    f"""
    **Configuration:**
    - **Activation Function:** `{activation}`
    - **Weight Initialization:** `{init_weights}`
    - **Normalization Type:** `{norm_type}`
    - **Optimizer:** `{optimizer_type}`
    - **Learning Rate:** `{learning_rate}`
    - **Skip Connection:** `{"Enabled" if use_skip else "Disabled"}`
    
    **Gradient Insights:**
    - Total Layers: **{num_layers}**
    - Minimum Gradient Norm: **{min(gradients):.4f}**
    - Maximum Gradient Norm: **{max(gradients):.4f}**
    - Gradient Norms displayed above for each layer.
    """
)
